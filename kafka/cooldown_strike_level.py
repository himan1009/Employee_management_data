# import time
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, count, when, lit, expr, lower, trim
# import os
# from datetime import datetime, timedelta

# # ✅ Control wait time here
# WAIT_SECONDS = 10  # Change to 20, 30, 40 etc as needed

# # ✅ Setup Spark Session
# jdbc_jar_path = "/home/himanshu/postgresql-42.7.5.jar"
# assert os.path.exists(jdbc_jar_path), "❌ JDBC JAR not found!"

# spark = SparkSession.builder \
#     .appName("CooldownSalaryUpdateOverwriteLoop") \
#     .config("spark.jars", jdbc_jar_path) \
#     .getOrCreate()

# spark.sparkContext.setLogLevel("WARN")

# # ✅ PostgreSQL Config
# pg_url = "jdbc:postgresql://localhost:5432/test_topic"
# pg_properties = {
#     "user": "postgres",
#     "password": "postgres",
#     "driver": "org.postgresql.Driver"
# }

# # ✅ Infinite loop
# while True:
#     print(f"⏳ Waiting for {WAIT_SECONDS} seconds before running cooldown update...")
#     for i in range(1, WAIT_SECONDS + 1):
#         time.sleep(1)
#         print(f"{i} second{'s' if i > 1 else ''} done...")

#     print("🚀 Running cooldown update...")

#     # ✅ Load previous strike_salary_status_table to preserve inactive employees
#     try:
#         previous_status_df = spark.read.jdbc(url=pg_url, table="strike_salary_status_table", properties=pg_properties)
#         inactive_df = previous_status_df.filter(lower(trim(col("status"))) == "inactive")
#         preserved_emp_ids = [row.emp_id for row in inactive_df.select("emp_id").distinct().collect()]
#         print("🧠 Preserved inactive emp_ids:", preserved_emp_ids)
#     except Exception as e:
#         print(f"⚠️ Could not load previous table: {e}")
#         inactive_df = None
#         preserved_emp_ids = []

#     # ✅ Load flagged_messages
#     flagged_df = spark.read.jdbc(url=pg_url, table="flagged_messages", properties=pg_properties) \
#         .select(col("sender_id").alias("emp_id"), col("date"))

#     # ✅ Load employee_db_salary
#     employee_salary_df = spark.read.jdbc(url=pg_url, table="employee_db_salary", properties=pg_properties)

#     # ✅ Filter strikes within last 30 days
#     now = datetime.now()
#     last_30_days = now - timedelta(days=30)
#     flagged_recent_df = flagged_df.filter(col("date") >= lit(last_30_days.strftime('%Y-%m-%d')))

#     # ✅ Count strikes per employee
#     strike_counts_df = flagged_recent_df.groupBy("emp_id").agg(count("date").alias("strike_count"))

#     # ✅ Remove inactive employees from update logic
#     if preserved_emp_ids:
#         strike_counts_df = strike_counts_df.filter(~col("emp_id").isin(preserved_emp_ids))
#         employee_salary_df = employee_salary_df.filter(~col("emp_id").isin(preserved_emp_ids))

#     # ✅ Join salary with active strike counts
#     combined_df = employee_salary_df.withColumnRenamed("salary", "base_salary") \
#         .join(strike_counts_df, on="emp_id", how="left") \
#         .fillna(0)

#     # ✅ Calculate salary after strikes
#     salary_cols = []
#     for i in range(1, 11):
#         factor = 0.9 ** i
#         salary_cols.append(
#             when(col("strike_count") >= i, expr(f"base_salary * {factor}"))
#             .otherwise(None).alias(f"salary_after_strike_{i}")
#         )

#     # ✅ Define status
#     status_col = when(col("strike_count") >= 10, lit("inactive")).otherwise(lit("active")).alias("status")

#     # ✅ Final DataFrame for active employees
#     active_df = combined_df.select(
#         col("emp_id"),
#         col("base_salary"),
#         *salary_cols,
#         status_col
#     )

#     # ✅ Merge preserved inactive rows
#     if inactive_df is not None:
#         # Ensure schema alignment
#         for i in range(1, 11):
#             col_name = f"salary_after_strike_{i}"
#             if col_name not in inactive_df.columns:
#                 inactive_df = inactive_df.withColumn(col_name, lit(None).cast("double"))
#         if "status" not in inactive_df.columns:
#             inactive_df = inactive_df.withColumn("status", lit("inactive"))

#         final_df = active_df.unionByName(inactive_df.select(active_df.columns))
#     else:
#         final_df = active_df

#     # ✅ Show result
#     final_df.select("emp_id", "status").show(truncate=False)
#     print(f"✅ Final rows count: {final_df.count()}")

#     # ✅ Write to strike_salary_status_table
#     final_df.write \
#         .format("jdbc") \
#         .option("url", pg_url) \
#         .option("dbtable", "strike_salary_status_table") \
#         .option("user", pg_properties["user"]) \
#         .option("password", pg_properties["password"]) \
#         .option("driver", pg_properties["driver"]) \
#         .mode("overwrite") \
#         .save()

#     print("✅ Cooldown update complete. Waiting for next cycle...")






import time
import psycopg2
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, lit, expr, lower, trim
import os
from datetime import datetime, timedelta

# ✅ Control wait time here
WAIT_SECONDS = 10  # Change to 20, 30, 40 etc as needed

# ✅ Setup Spark Session
jdbc_jar_path = "/home/ubuntu/postgresql-42.7.5.jar"
assert os.path.exists(jdbc_jar_path), "❌ JDBC JAR not found!"

spark = SparkSession.builder \
    .appName("CooldownSalaryUpdateOverwriteLoop") \
    .config("spark.jars", jdbc_jar_path) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ✅ PostgreSQL Config
pg_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
pg_properties = {
    "user": "test_user",
    "password": "test_user",
    "driver": "org.postgresql.Driver"
}

# ✅ Infinite loop
while True:
    print(f"⏳ Waiting for {WAIT_SECONDS} seconds before running cooldown update...")
    for i in range(1, WAIT_SECONDS + 1):
        time.sleep(1)
        print(f"{i} second{'s' if i > 1 else ''} done...")

    print("🚀 Running cooldown update...")

    # ✅ Load previous strike_salary_status_table to preserve inactive employees
    try:
        previous_status_df = spark.read.jdbc(url=pg_url, table="strike_salary_status_table", properties=pg_properties)
        inactive_df = previous_status_df.filter(lower(trim(col("status"))) == "inactive")
        preserved_emp_ids = [row.emp_id for row in inactive_df.select("emp_id").distinct().collect()]
        print("🧠 Preserved inactive emp_ids:", preserved_emp_ids)
    except Exception as e:
        print(f"⚠️ Could not load previous table: {e}")
        inactive_df = None
        preserved_emp_ids = []

    # ✅ Load flagged_messages
    flagged_df = spark.read.jdbc(url=pg_url, table="flagged_messages", properties=pg_properties) \
        .select(col("sender_id").alias("emp_id"), col("date"))

    # ✅ Load employee_db_salary
    employee_salary_df = spark.read.jdbc(url=pg_url, table="employee_db_salary", properties=pg_properties)

    # ✅ Filter strikes within last 30 days
    now = datetime.now()
    last_30_days = now - timedelta(days=30)
    flagged_recent_df = flagged_df.filter(col("date") >= lit(last_30_days.strftime('%Y-%m-%d')))

    # ✅ Count strikes per employee
    strike_counts_df = flagged_recent_df.groupBy("emp_id").agg(count("date").alias("strike_count"))

    # ✅ Remove inactive employees from update logic
    if preserved_emp_ids:
        strike_counts_df = strike_counts_df.filter(~col("emp_id").isin(preserved_emp_ids))
        employee_salary_df = employee_salary_df.filter(~col("emp_id").isin(preserved_emp_ids))

    # ✅ Join salary with active strike counts
    combined_df = employee_salary_df.withColumnRenamed("salary", "base_salary") \
        .join(strike_counts_df, on="emp_id", how="left") \
        .fillna(0)

    # ✅ Calculate salary after strikes
    salary_cols = []
    for i in range(1, 11):
        factor = 0.9 ** i
        salary_cols.append(
            when(col("strike_count") >= i, expr(f"base_salary * {factor}"))
            .otherwise(None).alias(f"salary_after_strike_{i}")
        )

    # ✅ Define status
    status_col = when(col("strike_count") >= 10, lit("inactive")).otherwise(lit("active")).alias("status")

    # ✅ Final DataFrame for active employees
    active_df = combined_df.select(
        col("emp_id"),
        col("base_salary"),
        *salary_cols,
        status_col
    )

    # ✅ ✅ ✅ NEWLY INACTIVE LOGIC ✅ ✅ ✅
    # Calculate newly inactive emp_ids (not previously inactive)
    current_inactive_df = active_df.filter(col("status") == "inactive")
    newly_inactive_df = current_inactive_df.filter(~col("emp_id").isin(preserved_emp_ids))
    newly_inactive_emp_ids = [row.emp_id for row in newly_inactive_df.select("emp_id").collect()]


    # ✅ Merge preserved inactive rows
    if inactive_df is not None:
        # Ensure schema alignment
        for i in range(1, 11):
            col_name = f"salary_after_strike_{i}"
            if col_name not in inactive_df.columns:
                inactive_df = inactive_df.withColumn(col_name, lit(None).cast("double"))
        if "status" not in inactive_df.columns:
            inactive_df = inactive_df.withColumn("status", lit("inactive"))

        final_df = active_df.unionByName(inactive_df.select(active_df.columns))
    else:
        final_df = active_df

    # ✅ Show result
    final_df.select("emp_id", "status").show(truncate=False)
    print(f"✅ Final rows count: {final_df.count()}")

    # ✅ Write to strike_salary_status_table
    final_df.write \
        .format("jdbc") \
        .option("url", pg_url) \
        .option("dbtable", "strike_salary_status_table") \
        .option("user", pg_properties["user"]) \
        .option("password", pg_properties["password"]) \
        .option("driver", pg_properties["driver"]) \
        .mode("overwrite") \
        .save()

    print("✅ Cooldown update complete in strike_salary_status_table.")

    # 🔁🔁🔁 START OF SQL-BASED INCREMENTAL UPDATE FOR employee_db_salary 🔁🔁🔁

    if newly_inactive_emp_ids:
        print("📌 Running direct SQL to mark employees as inactive in employee_db_salary:", newly_inactive_emp_ids)

        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host="localhost",
                port="5432",
                database="test_topic",
                user="test_user",
                password="test_user"
            )
            conn.autocommit = True
            cur = conn.cursor()

            # Create SQL string
            emp_id_str = ",".join([f"'{eid}'" for eid in newly_inactive_emp_ids])
            sql = f"""
                UPDATE employee_db_salary
                SET status = 'INACTIVE'
                WHERE emp_id IN ({emp_id_str});
            """

            cur.execute(sql)
            print("✅ employee_db_salary table updated successfully via SQL.")

            cur.close()
            conn.close()
        except Exception as e:
            print("❌ Error updating employee_db_salary:", str(e))
    else:
        print("✅ No new inactive employees to update in employee_db_salary.")

    # 🔁🔁🔁 END OF SQL-BASED INCREMENTAL UPDATE 🔁🔁🔁

    print("✅ Cooldown update complete. Waiting for next cycle...")
