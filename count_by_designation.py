import sys
import traceback
from pyspark.context import SparkContext
from pyspark.sql.functions import col, lower, count, when, lit
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job

# Step 1: Glue job args
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Step 2: JDBC source and target info
jdbc_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
input_table = "employee_db_salary"
output_table = "count_by_designation"
db_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}

try:
    print("Reading input data from PostgreSQL table...")
    df = spark.read.jdbc(url=jdbc_url, table=input_table, properties=db_properties)

    print("Filtering active employees and grouping by designation...")
    active_employees_df = (
        df.filter(lower(col("status")) == "active")
        .groupBy(
            when(col("designation").isNull(), lit("UNKNOWN"))
            .otherwise(col("designation"))
            .alias("designation")
        )
        .agg(count("emp_id").alias("active_emp_count"))
        .orderBy("active_emp_count", ascending=False)
    )

    if active_employees_df.count() > 0:
        print("Writing designation-wise active employee count to database...")
        active_employees_df.write \
            .mode("overwrite") \
            .jdbc(url=jdbc_url, table=output_table, properties=db_properties)
        print("✅ Write successful to table 'count_by_designation'.")
    else:
        print("⚠️ No active employees found. Nothing written.")

except Exception:
    print("❌ Job failed due to error:")
    print(traceback.format_exc())
    raise

# Step 3: Commit job
job.commit()
