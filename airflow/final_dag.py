from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.operators.python import ShortCircuitOperator
from datetime import datetime, timedelta
import os

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# === Corrected Flag Check Logic Using Execution Context ===

def run_monthly_job(**kwargs):
    exec_date = kwargs["execution_date"]
    if exec_date.day != 8:
        return False
    month_flag = exec_date.strftime("/tmp/80_leave_report_%Y-%m.flag")
    if os.path.exists(month_flag):
        return False
    open(month_flag, "w").close()
    return True

def run_yearly_job(**kwargs):
    exec_date = kwargs["execution_date"]
    if not (exec_date.day == 8 and exec_date.month == 5):
        return False
    year_flag = f"/tmp/leave_quota_done_{exec_date.year}.flag"
    if os.path.exists(year_flag):
        return False
    open(year_flag, "w").close()
    return True

# === DAG Definition ===

with DAG(
    dag_id="employee_etl_all_tasks",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 7 * * *",  # Daily at 7 AM UTC
    catchup=False,
    tags=["glue", "unified-etl"]
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    # === Daily Glue Jobs ===

    emp_data = GlueJobOperator(
        task_id="employee_data_job",
        job_name="poc-bootcamp-group1-emp-data",
        aws_conn_id="aws_default"
    )

    emp_timeframe = GlueJobOperator(
        task_id="employee_timeframe_job",
        job_name="poc-bootcamp-group1-emp-timeframe-data",
        aws_conn_id="aws_default"
    )

    count_by_designation = GlueJobOperator(
        task_id="designation_count_job",
        job_name="poc-bootcamp-group1-countbydesignation",
        aws_conn_id="aws_default"
    )

    leave_data = GlueJobOperator(
        task_id="leave_data_job",
        job_name="poc-bootcamp-group1-leave-data",
        aws_conn_id="aws_default"
    )

    eight_percent_leave = GlueJobOperator(
        task_id="eight_percent_leave_checker",
        job_name="poc-bootcamp-Group1-8-percent",
        aws_conn_id="aws_default"
    )

    # === Monthly Logic ===

    check_monthly = ShortCircuitOperator(
        task_id="check_monthly_flag",
        python_callable=run_monthly_job,
        provide_context=True
    )

    monthly_leave_report = GlueJobOperator(
        task_id="monthly_80_percent_leave_checker",
        job_name="poc-bootcamp-Group1-80-percent",
        aws_conn_id="aws_default",
        retries=3,
        retry_delay=timedelta(minutes=10),
        retry_exponential_backoff=True,
        max_retry_delay=timedelta(minutes=30),
    )

    # === Yearly Logic ===

    check_yearly = ShortCircuitOperator(
        task_id="check_yearly_flag",
        python_callable=run_yearly_job,
        provide_context=True
    )

    leave_quota = GlueJobOperator(
        task_id="leave_quota_loader",
        job_name="poc-bootcamp-group1-leave-quota-data",
        aws_conn_id="aws_default"
    )

    leave_calendar = GlueJobOperator(
        task_id="leave_calendar_loader",
        job_name="poc-bootcamp-group1-leave-calender-data",
        aws_conn_id="aws_default"
    )

    # === DAG Flow ===

    # Daily path
    start >> emp_data >> emp_timeframe
    emp_timeframe >> [count_by_designation, leave_data]
    leave_data >> eight_percent_leave >> end

    # Monthly path (corrected logic)
    start >> check_monthly
    [check_monthly, leave_data] >> monthly_leave_report >> end
    check_monthly >> end  # if skipped

    # Yearly path
    start >> check_yearly
    check_yearly >> [leave_quota, leave_calendar]
    leave_quota >> end
    leave_calendar >> end
    check_yearly >> end
