import sys
import boto3
from datetime import datetime
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import col

# Initialize Spark and Glue
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

# Date setup
today = datetime.utcnow().strftime("%Y-%m-%d")

# S3 Paths
bucket = "poc-bootcamp-capstone-group1"
raw_prefix = "poc-bootcamp-group1-bronze/emp_data_qus1/raw/"
processed_output_path = f"s3://{bucket}/poc-bootcamp-group1-bronze/emp_data_qus1/processed/data_processed_{today}.csv"

# JDBC Config
jdbc_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
db_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}
output_table = "employee_db"

# S3 Client
s3 = boto3.client('s3')

# Step 1: List raw CSV files
raw_files = s3.list_objects_v2(Bucket=bucket, Prefix=raw_prefix).get("Contents", [])
csv_files = [obj["Key"] for obj in raw_files if obj["Key"].endswith(".csv")]

if not csv_files:
    print("No CSV files found in raw folder.")
    sys.exit(0)

print(f" Found {len(csv_files)} CSV files.")

# Step 2: Merge CSVs
raw_paths = [f"s3://{bucket}/{key}" for key in csv_files]
df_raw = spark.read.option("header", "true").csv(raw_paths)

# Save processed backup to processed folder
df_raw.coalesce(1).write.mode("overwrite").option("header", "true").csv(processed_output_path)
print(f" Merged raw CSV backed up to: {processed_output_path}")

# Delete raw files after backup
for key in csv_files:
    s3.delete_object(Bucket=bucket, Key=key)
    print(f"️ Deleted raw file: {key}")

# Step 3: Load from processed CSV
df_processed = spark.read.option("header", "true").csv(processed_output_path)

# Step 4: Clean and transform
df_cleaned = df_processed.select(
    col("emp_id").cast("string"),
    col("age").cast("int"),
    col("name").cast("string")
).dropna().filter(col("age") > 0).dropDuplicates(["emp_id", "age", "name"])

# Step 5: Write to PostgreSQL
if not df_cleaned.rdd.isEmpty():
    df_cleaned.write.mode("append").jdbc(url=jdbc_url, table=output_table, properties=db_properties)
    print(f"Final cleaned employee data written to PostgreSQL table: {output_table}")
else:
    print("No valid rows to write after cleaning.")

print(" Glue job completed successfully.")
