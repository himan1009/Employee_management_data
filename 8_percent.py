import sys
from datetime import datetime
import traceback
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import *
from pyspark.sql.types import DateType

# Init Glue job 
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark._jsc.hadoopConfiguration().set("spark.jars", "/home/ubuntu/postgresql-42.7.2.jar")

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# JDBC config 
jdbc_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
db_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}
calendar_table = "leave_calendar_data"
leave_table = "leave_data"
output_table = "eight_percent"

# Get parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME', 'today', 'start_of_year', 'end_of_year', 'CURRENT_YEAR'
])
CURRENT_YEAR = int(args['CURRENT_YEAR'])
today = datetime.strptime(args['today'], "%Y-%m-%d")
JAN_1 = datetime.strptime(args['start_of_year'], "%Y-%m-%d")
end_of_year = datetime.strptime(args['end_of_year'], "%Y-%m-%d")

today_str = today.strftime("%Y-%m-%d")
end_of_year_str = end_of_year.strftime("%Y-%m-%d")

print(f"Running for date range: {today_str} to {end_of_year_str}")

try:
    # Read from PostgreSQL
    calendar_df = spark.read.jdbc(url=jdbc_url, table=calendar_table, properties=db_properties) \
                          .withColumn("date", col("date").cast(DateType()))
    leave_df = spark.read.jdbc(url=jdbc_url, table=leave_table, properties=db_properties) \
                        .withColumn("leave_date", col("date").cast(DateType()))

    # Clean leave data 
    leave_df = leave_df.filter((col("status") == 'ACTIVE') & (col("leave_date") >= lit(JAN_1))) \
                       .select("emp_id", "leave_date", "status").dropDuplicates(["emp_id", "leave_date"])
    print("Loaded leave entries:", leave_df.count())

    # Generate date range 
    date_range = spark.sql(f"SELECT explode(sequence(to_date('{today_str}'), to_date('{end_of_year_str}'))) AS date")
    weekends = date_range.withColumn("day_of_week", dayofweek("date")).filter(col("day_of_week").isin([1, 7])).select("date")
    non_working_days = calendar_df.select("date").union(weekends).distinct()
    working_days = date_range.join(non_working_days, on="date", how="left_anti")

    total_working_days = working_days.count()
    print("Total working days left:", total_working_days)

    if total_working_days == 0:
        print("No valid working days. Exiting.")
    else:
        # Future valid ACTIVE leaves 
        future_leaves = leave_df \
            .filter((col("leave_date") >= lit(today_str)) & (col("leave_date") <= lit(end_of_year_str))) \
            .join(working_days, leave_df.leave_date == working_days.date, "inner") \
            .select("emp_id", "leave_date")

        print("Valid future ACTIVE working-day leaves:", future_leaves.count())

        # Leave % Flagging 
        leave_counts = future_leaves.groupBy("emp_id") \
            .agg(countDistinct("leave_date").alias("upcoming_leaves_count"))

        flagged = leave_counts.withColumn(
            "leave_percent", (col("upcoming_leaves_count") / lit(total_working_days)) * 100
        ).withColumn(
            "flagged", when(col("leave_percent") > 8, "Yes").otherwise("No")
        )

        # Final flagged employees 
        final_flagged = flagged.filter(col("flagged") == "Yes") \
                               .select("emp_id", "upcoming_leaves_count")

        print("Flagged employee count:", final_flagged.count())
        final_flagged.show()

        # Write to PostgreSQL
        if not final_flagged.rdd.isEmpty():
            final_flagged.write.mode("overwrite").jdbc(
                url=jdbc_url, table=output_table, properties=db_properties
            )
            print("Written flagged report to PostgreSQL table:", output_table)
        else:
            print("No flagged employees found. Skipping write.")

except Exception:
    print("Job failed:")
    print(traceback.format_exc())
    raise

job.commit()
