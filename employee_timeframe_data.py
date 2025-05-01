from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from py4j.protocol import Py4JJavaError
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType
import sys
import boto3
import re

# === Glue Setup ===
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark.sparkContext.setLogLevel("ERROR")

# === Define Schema Manually ===
manual_schema = StructType([
   StructField("emp_id", StringType(), True),
   StructField("designation", StringType(), True),
   StructField("start_date", DateType(), True),
   StructField("end_date", DateType(), True),
   StructField("salary", DoubleType(), True),
   StructField("status", StringType(), True)
])

# === S3 Paths ===
RAW_PATH = "s3://poc-bootcamp-capstone-group1/poc-bootcamp-group1-bronze/emp_timeframe_data_qus2/raw/"
STAGING_PATH = "s3://poc-bootcamp-capstone-group1/poc-bootcamp-group1-bronze/emp_timeframe_data_qus2/processed/"
SILVER_PATH = "s3://poc-bootcamp-capstone-group1/poc-bootcamp-group1-silver/emp-timeframe-data/"
PROCESSED_RAW_PATH = "s3://poc-bootcamp-capstone-group1/poc-bootcamp-group1-bronze/emp_timeframe_data_qus2/processed_raw/"

# === Step 1: Read raw data
df = spark.read.option("header", True).csv(RAW_PATH)
df = df.withColumn("salary", F.col("salary").cast("double"))
df = df.withColumn("start_date", F.to_date(F.from_unixtime(F.col("start_date").cast("long"))))
df = df.withColumn("end_date", F.to_date(F.from_unixtime(F.col("end_date").cast("long"))))
df.show()

# === Step 2: Deduplicate by (emp_id, start_date) using highest salary
w_dedupe = Window.partitionBy("emp_id", "start_date").orderBy(F.col("salary").desc())
df = df.withColumn("rn", F.row_number().over(w_dedupe)).filter(F.col("rn") == 1).drop("rn")

# === Step 3: Fill end_date and mark ACTIVE
w_desc = Window.partitionBy("emp_id").orderBy(F.col("start_date").desc())
df = df.withColumn("is_latest", F.row_number().over(w_desc) == 1)

w_lead = Window.partitionBy("emp_id").orderBy("start_date")
df = df.withColumn("next_start_date", F.lead("start_date").over(w_lead))
df = df.withColumn("end_date", F.when(F.col("is_latest"), None).otherwise(F.col("next_start_date"))) \
       .drop("next_start_date", "is_latest")

df = df.withColumn("status", F.when(F.col("end_date").isNull(), "ACTIVE").otherwise("INACTIVE"))

# === Step 4: Save to staging
df.write.mode("overwrite").parquet(STAGING_PATH)

# === Step 5: Load Silver or create new
s3 = boto3.client("s3")
bucket = "poc-bootcamp-capstone-group1"
prefix = "poc-bootcamp-group1-silver/emp-timeframe-data/"
exists = "Contents" in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

if not exists:
    df.write.mode("overwrite").parquet(SILVER_PATH)
else:
    try:
        silver_df = spark.read.schema(manual_schema).parquet(SILVER_PATH)
    except Py4JJavaError:
        silver_df = spark.createDataFrame([], schema=manual_schema)

    processed_df = spark.read.schema(manual_schema).parquet(STAGING_PATH)
    active_df = silver_df.filter(F.col("status") == "ACTIVE")
    inactive_df = silver_df.filter(F.col("status") == "INACTIVE")

    continuity = processed_df.groupBy("emp_id").agg(F.min("start_date").alias("new_start_date"))

    updated_active = active_df.alias("old") \
        .join(continuity.alias("new"), "emp_id") \
        .filter(F.col("new.new_start_date") >= F.col("old.start_date")) \
        .withColumn("end_date", F.col("new.new_start_date")) \
        .withColumn("status", F.lit("INACTIVE")) \
        .select("old.emp_id", "old.designation", "old.start_date", "end_date", "old.salary", "status")

    untouched_active = active_df.join(continuity.select("emp_id"), "emp_id", "left_anti")
    new_emps = processed_df.join(silver_df.select("emp_id").distinct(), "emp_id", "left_anti")

    # Merge
    columns = ["emp_id", "designation", "start_date", "end_date", "salary", "status"]
    final_df = inactive_df.select(columns) \
        .union(updated_active.select(columns)) \
        .union(untouched_active.select(columns)) \
        .union(processed_df.select(columns)) \
        .union(new_emps.select(columns)) \
        .dropDuplicates(columns)

    # Reapply logic: dedup, end_date, status
    w_dedupe2 = Window.partitionBy("emp_id", "start_date").orderBy(F.col("salary").desc())
    final_df = final_df.withColumn("row_num", F.row_number().over(w_dedupe2)).filter(F.col("row_num") == 1).drop("row_num")

    w_desc2 = Window.partitionBy("emp_id").orderBy(F.col("start_date").desc())
    final_df = final_df.withColumn("is_latest", F.row_number().over(w_desc2) == 1)

    final_df = final_df.withColumn("next_start_date", F.lead("start_date").over(w_lead))
    final_df = final_df.withColumn("end_date", F.when(F.col("is_latest"), None).otherwise(F.col("next_start_date"))) \
                       .drop("next_start_date", "is_latest")

    final_df = final_df.withColumn("status", F.when(F.col("end_date").isNull(), "ACTIVE").otherwise("INACTIVE"))

    w_final = Window.partitionBy("emp_id", "start_date", "end_date").orderBy(F.col("salary").desc())
    final_df = final_df.withColumn("rn_final", F.row_number().over(w_final)).filter(F.col("rn_final") == 1).drop("rn_final")

    final_df.write.mode("overwrite").parquet(SILVER_PATH)
    print("💽 Final Silver Table updated!")

# === Step 6: Archive raw and clean staging
def extract_bucket_prefix(s3_path):
    match = re.match(r's3://([^/]+)/(.+)', s3_path)
    return match.groups() if match else (None, None)

def move_raw_files(bucket, raw_prefix, archive_prefix):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=raw_prefix).get("Contents", [])
    for obj in objs:
        key = obj["Key"]
        if not key.endswith('/'):
            s3.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': key}, Key=key.replace(raw_prefix, archive_prefix, 1))
            s3.delete_object(Bucket=bucket, Key=key)
            print(f"📦 Archived {key}")

def clean_staging(bucket, prefix):
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
    for obj in objs:
        key = obj["Key"]
        if not key.endswith('/'):
            s3.delete_object(Bucket=bucket, Key=key)
            print(f"🧹 Deleted {key}")

raw_bucket, raw_prefix = extract_bucket_prefix(RAW_PATH)
staging_bucket, staging_prefix = extract_bucket_prefix(STAGING_PATH)
_, processed_raw_prefix = extract_bucket_prefix(PROCESSED_RAW_PATH)

if raw_bucket:
    move_raw_files(raw_bucket, raw_prefix, processed_raw_prefix)

if staging_bucket:
    clean_staging(staging_bucket, staging_prefix)

print("✅ Glue Job Finished Successfully.")
