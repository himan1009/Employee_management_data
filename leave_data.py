import sys
from datetime import datetime
import traceback
from pyspark.sql.types import StructType, StructField, StringType, DateType, LongType
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job
from pyspark.sql.functions import (
    col, lit, current_timestamp, row_number, when, sum, year, month, dayofweek
)
from pyspark.sql import Window

# Initialize Glue job
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# S3 Input Path
bucket = "poc-bootcamp-capstone-group1"
bronze_path = f"s3://{bucket}/poc-bootcamp-group1-bronze/emp_leave_data/"

# PostgreSQL Configuration
jdbc_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
db_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}
emp_salary_table = "employee_db_salary"
final_output_table = "leave_data"

# Schema for leave data
leave_schema = StructType([
    StructField("emp_id", StringType(), True),
    StructField("date", DateType(), True),
    StructField("status", StringType(), True)
])

try:
    # Step 1: Read leave data from S3
    leave_data = spark.read.schema(leave_schema).option("header", True).csv(bronze_path)
    leave_data = leave_data.filter(~dayofweek(col("date")).isin([1, 7]))  # Remove weekends

    # Step 2: Read ACTIVE employees from PostgreSQL
    emp_time_df = spark.read.jdbc(url=jdbc_url, table=emp_salary_table, properties=db_properties) \
                           .filter(col("status") == "ACTIVE") \
                           .select("emp_id").distinct()
    print("Active employee count:", emp_time_df.count())

    # Step 3: Join leave with active employee list
    leave_data = leave_data.join(emp_time_df, on="emp_id", how="left_semi")

    # Step 4: Add metadata
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    leave_data = leave_data.withColumn("ingest_date", lit(today_str)) \
                           .withColumn("ingest_timestamp", current_timestamp())

    # Step 5: Status resolution
    status_df = leave_data.withColumn("is_active", when(col("status") == "ACTIVE", 1).otherwise(0)) \
                          .withColumn("is_cancelled", when(col("status") == "CANCELLED", 1).otherwise(0))

    status_summary = status_df.groupBy("emp_id", "date").agg(
        sum("is_active").alias("active_count"),
        sum("is_cancelled").alias("cancelled_count")
    )

    final_status = status_summary.withColumn(
        "final_status",
        when(col("cancelled_count") >= col("active_count"), lit("CANCELLED")).otherwise(lit("ACTIVE"))
    )

    # Step 6: Keep only final rows
    filtered_df = leave_data.join(
        final_status.select("emp_id", "date", "final_status"),
        on=["emp_id", "date"]
    ).filter(col("status") == col("final_status"))

    # Step 7: Deduplicate
    window_spec = Window.partitionBy("emp_id", "date").orderBy(col("ingest_timestamp").desc())
    deduped_df = filtered_df.withColumn("row_num", row_number().over(window_spec)) \
                            .filter(col("row_num") == 1) \
                            .drop("row_num", "final_status") \
                            .withColumn("year", year(col("date"))) \
                            .withColumn("month", month(col("date"))) \
                            .withColumn("emp_id", col("emp_id").cast(LongType()))

    # Step 8: Write to PostgreSQL
    if not deduped_df.rdd.isEmpty():
        deduped_df.write \
            .mode("overwrite") \
            .jdbc(url=jdbc_url, table=final_output_table, properties=db_properties)
        print(f"Leave data written to PostgreSQL table '{final_output_table}'.")
    else:
        print("No leave data to write. Skipping DB write.")

except Exception:
    print("Job failed due to error:")
    print(traceback.format_exc())
    raise

# Finalize
job.commit()
