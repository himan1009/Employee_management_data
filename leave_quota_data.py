import sys
import traceback
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import col, max

# Glue job args
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# Initialize Glue + Spark
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Input S3 path
INPUT_PATH = "s3://poc-bootcamp-capstone-group1/poc-bootcamp-group1-bronze/emp_leave_quota/"

# Schema
schema = StructType([
    StructField("emp_id", IntegerType(), True),
    StructField("leave_quota", IntegerType(), True),
    StructField("year", IntegerType(), True)
])

# JDBC connection info
jdbc_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
table_name = "leave_quota_data"
db_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}

try:
    # Read data
    df = spark.read.option("header", True).schema(schema).csv(INPUT_PATH)
    df = df.filter(
            (col("emp_id").isNotNull()) &
            (col("leave_quota").isNotNull()) &
            (col("year").isNotNull())
        )

    if df.rdd.isEmpty():
        print("No data found. Skipping write.")
    else:
        # Clean data
        df = df.groupBy("emp_id", "year").agg(max("leave_quota").alias("leave_quota"))

        # Write to PostgreSQL table
        df.write \
            .mode("append") \
            .jdbc(url=jdbc_url, table=table_name, properties=db_properties)

        print(f"Successfully written to table '{table_name}'.")

except Exception:
    print("Job failed due to error:")
    print(traceback.format_exc())
    raise

finally:
    job.commit()
