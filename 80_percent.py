import sys
import boto3
import traceback
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql.functions import col, to_date, countDistinct, lit
from pyspark.sql.types import StringType

# ========== Init ==========
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ========== Config ==========
today = datetime(2024, 12, 31)
start_of_year = date(2024, 1, 1)
run_date_str = today.strftime("%Y-%m-%d")
current_year = today.year

bucket = "poc-bootcamp-capstone-group1"
output_prefix = "poc-bootcamp-group1-gold/emp_80_percent_flagged/"

jdbc_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
db_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}
leave_table = "leave_data"
quota_table = "leave_quota_data"
final_output_table = "eighty_percent"

s3_client = boto3.client("s3")

try:
    print("Step 1: Read Leave & Quota Data from PostgreSQL")

    leave_df = spark.read.jdbc(url=jdbc_url, table=leave_table, properties=db_properties) \
        .withColumn("date", to_date("date"))

    quota_df = spark.read.jdbc(url=jdbc_url, table=quota_table, properties=db_properties) \
        .withColumnRenamed("year", "quota_year") \
        .withColumn("emp_id", col("emp_id").cast(StringType()))

    print("Step 2: Filter ACTIVE leave records for current year")
    valid_leaves = leave_df.filter(
        (col("status") == "ACTIVE") &
        (col("date") >= lit(start_of_year)) &
        (col("date") <= lit(today))
    ).select("emp_id", "date").distinct().withColumn("emp_id", col("emp_id").cast(StringType()))

    print("Step 3: Count distinct leave days")
    leaves_taken = valid_leaves.groupBy("emp_id").agg(countDistinct("date").alias("leaves_taken"))

    print("Step 4: Join with quota and calculate usage")
    leave_usage = leaves_taken.join(quota_df, on="emp_id", how="inner") \
        .filter(col("quota_year") == current_year) \
        .filter(col("leave_quota") > 0) \
        .withColumn("leave_percent", (col("leaves_taken") / col("leave_quota")) * 100) \
        .withColumn("flagged", lit("Yes")) \
        .filter(col("leave_percent") > 80)

    flagged_count = leave_usage.count()
    print("Total flagged employees:", flagged_count)

    if flagged_count == 0:
        print("No employees exceeded 80% usage. Skipping report generation.")
    else:
        print("Step 5: Avoid duplicates and write TXT reports")
        existing_keys = set()
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=output_prefix)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".txt"):
                emp = key.split("/")[-1].split("_")[0]
                existing_keys.add(emp)

        for row in leave_usage.collect():
            emp_id = row["emp_id"]
            if emp_id in existing_keys:
                continue

            content = (
                f"Employee ID: {emp_id}\n"
                f"Leave Taken: {row['leaves_taken']}\n"
                f"Leave Quota: {row['leave_quota']}\n"
                f"Usage: {row['leave_percent']:.2f}%\n"
                f"Report Date: {run_date_str}\n"
            )
            key = f"{output_prefix}{emp_id}_report.txt"
            s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
            print(f"✅ Report written for: {emp_id}")

        print("Step 6: Write final DataFrame to PostgreSQL")

        final_df = leave_usage.select(
            "emp_id",
            "leaves_taken",
            "leave_quota",
            col("quota_year").alias("year"),
            "leave_percent",
            "flagged"
        )

        final_df.write.mode("overwrite").jdbc(
            url=jdbc_url,
            table=final_output_table,
            properties=db_properties
        )

        print(f"✅ Final summary written to PostgreSQL table: {final_output_table}")

except Exception as e:
    print("❌ Job failed with error:")
    print(traceback.format_exc())
    raise

# ========== Finish ==========
job.commit()
