import sys
import traceback
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql.functions import col, year

# Get job arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

# Initialize contexts
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Set path to PostgreSQL JDBC JAR
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# S3 source
bronze_path = "s3://poc-bootcamp-capstone-group1/poc-bootcamp-group1-bronze/emp_leave_calender/"

# Define schema
schema = StructType([
    StructField("reason", StringType(), True),
    StructField("date", TimestampType(), True)
])

# PostgreSQL config
jdbc_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
table_name = "leave_calendar_data"
db_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}

try:
    # Read from S3
    df = spark.read \
        .format("csv") \
        .schema(schema) \
        .option("header", True) \
        .load(bronze_path)

    if df.rdd.isEmpty():
        print("No data found in input. Skipping write.")
    else:
        # Clean and transform
        df = df.dropDuplicates()
        df = df.withColumn("year", year(col("date")))

        # Write to PostgreSQL
        df.write \
            .mode("append") \
            .jdbc(url=jdbc_url, table=table_name, properties=db_properties)

        print(f"Successfully written {df.count()} records to table '{table_name}'.")

except Exception as e:
    print("Job failed due to error:")
    print(traceback.format_exc())
    raise

finally:
    job.commit()
