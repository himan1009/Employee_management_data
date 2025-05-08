import sys
import traceback
from pyspark.context import SparkContext
from pyspark.sql.functions import col, lower, count, when, lit
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job

# Step 1: Glue setup
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Step 2: JDBC setup
jdbc_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
input_table = "employee_db_salary"
output_table = "count_by_designation"
db_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}

try:
    print("Reading employee data from PostgreSQL...")
    df = spark.read.jdbc(url=jdbc_url, table=input_table, properties=db_properties)

    print("Calculating active employee count by designation...")
    result_df = (
        df.filter(lower(col("status")) == "active")
        .withColumn("designation", when(col("designation").isNull(), lit("UNKNOWN")).otherwise(col("designation")))
        .groupBy("designation")
        .agg(count("emp_id").alias("active_emp_count"))
        .orderBy("active_emp_count", ascending=False)
    )

    print("Writing result to PostgreSQL...")
    result_df.write.mode("overwrite").jdbc(url=jdbc_url, table=output_table, properties=db_properties)
    print("Write successful to table:", output_table)

except Exception:
    print("Job failed due to an error:")
    print(traceback.format_exc())
    raise

job.commit()
