import sys
import shutil
from io import StringIO
import os
import boto3
import re
import psycopg2
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, from_unixtime, to_date, when, row_number, lit, min as min_, lead
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

# Register PostgreSQL JDBC driver JAR from S3 if not already passed via job parameter
# NOTE: The correct way in Glue is to use --extra-jars while submitting the job
# Example: --extra-jars s3://your-bucket/jars/postgresql-42.7.2.jar
spark.sparkContext.setLogLevel("ERROR")

POSTGRES_URL = "jdbc:postgresql://3.221.182.234:5432/test_topic"
POSTGRES_PROPERTIES = {"user": "test_user", "password": "test_user", "driver": "org.postgresql.Driver"}
PSYCOPG2_CONN = dict(dbname="test_topic", user="test_user", password="test_user", host="3.221.182.234", port="5432")

STAGING_TABLE = "staged_employees"
FINAL_TABLE = "employee_db_salary"
BACKUP_TABLE = "employee_db_salary_backup"
TEMP_LOCAL_PATH = "/tmp/final_pg_upload"

# S3 Paths for raw and archive 
RAW_PATH = "s3://poc-bootcamp-capstone-group1/poc-bootcamp-group1-bronze/emp_timeframe_data_qus2/raw/"
PROCESSED_RAW_PATH = "s3://poc-bootcamp-capstone-group1/poc-bootcamp-group1-bronze/emp_timeframe_data_qus2/processed_raw/"

# Manual Schema 
schema = StructType([
    StructField("emp_id", StringType(), True),
    StructField("designation", StringType(), True),
    StructField("start_date", DateType(), True),
    StructField("end_date", DateType(), True),
    StructField("salary", DoubleType(), True),
    StructField("status", StringType(), True)
])

# Read Raw File 
df = spark.read.option("header", True).csv(RAW_PATH)
df = df.withColumn("start_date", to_date(from_unixtime(col("start_date").cast("long"))))
df = df.withColumn("end_date", to_date(from_unixtime(col("end_date").cast("long"))))
df = df.withColumn("salary", col("salary").cast("double"))

# Deduplication 
w1 = Window.partitionBy("emp_id", "start_date", "end_date").orderBy(col("salary").desc())
df = df.withColumn("rn1", row_number().over(w1)).filter(col("rn1") == 1).drop("rn1")

w2 = Window.partitionBy("emp_id").orderBy(col("salary").desc())
df_with_nulls = df.filter(col("end_date").isNull()).withColumn("rn2", row_number().over(w2)).filter(col("rn2") == 1).drop("rn2")
df_non_nulls = df.filter(col("end_date").isNotNull())
df = df_with_nulls.unionByName(df_non_nulls)
df = df.withColumn("status", when(col("end_date").isNull(), "ACTIVE").otherwise("INACTIVE"))

# Write to staging 
df.write.mode("overwrite").jdbc(POSTGRES_URL, STAGING_TABLE, properties=POSTGRES_PROPERTIES)

# Load existing data 
tables_df = spark.read.jdbc(POSTGRES_URL, "information_schema.tables", properties=POSTGRES_PROPERTIES)
exists = tables_df.filter(col("table_name") == FINAL_TABLE).count() > 0

if not exists:
    df.write.mode("overwrite").jdbc(POSTGRES_URL, FINAL_TABLE, properties=POSTGRES_PROPERTIES)
else:
    silver_df = spark.read.jdbc(POSTGRES_URL, FINAL_TABLE, properties=POSTGRES_PROPERTIES)
    processed_df = spark.read.jdbc(POSTGRES_URL, STAGING_TABLE, properties=POSTGRES_PROPERTIES)

    active_df = silver_df.filter(col("status") == "ACTIVE")
    inactive_df = silver_df.filter(col("status") == "INACTIVE")
    continuity_dates = processed_df.groupBy("emp_id").agg(min_("start_date").alias("new_start_date"))

    updated_old = active_df.alias("old") \
        .join(continuity_dates.alias("new"), "emp_id") \
        .filter(col("new.new_start_date") >= col("old.start_date")) \
        .withColumn("end_date", col("new.new_start_date")) \
        .withColumn("status", lit("INACTIVE")) \
        .select("old.emp_id", "old.designation", "old.start_date", "end_date", "old.salary", "status")

    untouched_active = active_df.join(continuity_dates.select("emp_id"), "emp_id", "left_anti")
    new_emps = processed_df.join(silver_df.select("emp_id").distinct(), "emp_id", "left_anti")

    columns = ["emp_id", "designation", "start_date", "end_date", "salary", "status"]
    final_df = inactive_df.select(columns) \
        .union(updated_old.select(columns)) \
        .union(untouched_active.select(columns)) \
        .union(processed_df.select(columns)) \
        .union(new_emps.select(columns)) \
        .dropDuplicates(columns)

    w_dedupe = Window.partitionBy("emp_id", "start_date").orderBy(col("salary").desc())
    final_df = final_df.withColumn("row_num", row_number().over(w_dedupe)).filter(col("row_num") == 1).drop("row_num")

    w_desc = Window.partitionBy("emp_id").orderBy(col("start_date").desc())
    final_df = final_df.withColumn("is_latest", row_number().over(w_desc) == 1)
    w_ordered = Window.partitionBy("emp_id").orderBy("start_date")
    final_df = final_df.withColumn("next_start_date", lead("start_date").over(w_ordered))
    final_df = final_df.withColumn("end_date", when(col("is_latest"), None).otherwise(col("next_start_date"))) \
                       .drop("next_start_date", "is_latest")
    final_df = final_df.withColumn("status", when(col("end_date").isNull(), "ACTIVE").otherwise("INACTIVE"))

    # Backup table
    conn = psycopg2.connect(**PSYCOPG2_CONN)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS employee_db_salary_backup;")
    cur.execute("CREATE TABLE employee_db_salary_backup AS TABLE employee_db_salary;")

    # Delete matching rows using temp table
    keys = final_df.select("emp_id", "start_date").distinct().collect()
    key_tuples = [(row["emp_id"], row["start_date"].strftime("%Y-%m-%d")) for row in keys]
    if key_tuples:
        cur.execute("DROP TABLE IF EXISTS delete_keys;")
        cur.execute("CREATE TEMP TABLE delete_keys (emp_id TEXT, start_date DATE);")

        key_buf = StringIO("".join([f"{e},{d}\n" for e, d in key_tuples]))
        key_buf.seek(0)
        cur.copy_expert("COPY delete_keys FROM STDIN WITH CSV DELIMITER ',' NULL '';", key_buf)

        cur.execute("""
            DELETE FROM employee_db_salary
            USING delete_keys
            WHERE employee_db_salary.emp_id = delete_keys.emp_id
              AND employee_db_salary.start_date = delete_keys.start_date;
        """)

    # Copy data into PostgreSQL
    final_df.coalesce(1).write.option("header", False).option("delimiter", ",").mode("overwrite").save(TEMP_LOCAL_PATH)

    # Convert Spark DataFrame to Pandas and write directly to buffer
    pandas_df = final_df.toPandas()
    buffer = StringIO()
    pandas_df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    cur.copy_expert("""
        COPY employee_db_salary (emp_id, designation, start_date, end_date, salary, status)
        FROM STDIN WITH CSV DELIMITER ',' NULL '';
    """, buffer)

    conn.commit()
    cur.close()
    conn.close()
    print("Final table updated using psycopg2 (delete + insert).")

# Archive raw file 
s3 = boto3.client("s3")
raw_bucket = RAW_PATH.split("/")[2]
raw_prefix = "/".join(RAW_PATH.split("/")[3:])
archive_prefix = raw_prefix.replace("raw/", "processed_raw/")

objs = s3.list_objects_v2(Bucket=raw_bucket, Prefix=raw_prefix).get("Contents", [])
for obj in objs:
    key = obj["Key"]
    if not key.endswith('/'):
        s3.copy_object(Bucket=raw_bucket, CopySource={'Bucket': raw_bucket, 'Key': key}, Key=archive_prefix + key.split('/')[-1])
        s3.delete_object(Bucket=raw_bucket, Key=key)
        print(f"Archived {key}")
