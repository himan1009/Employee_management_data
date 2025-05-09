from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='kafka_cooldown_trigger_only',
    default_args=default_args,
    start_date=datetime(2025, 5, 7),
    schedule_interval='0 8 * * *',
    catchup=False,
    tags=['kafka', 'cooldown'],
) as dag:

    start = EmptyOperator(task_id='start')

    run_cooldown = BashOperator(
        task_id='run_cooldown',
        bash_command=(
            "source ~/airflow_venv/bin/activate && "
            "python3 /home/ubuntu/kafka_codes/KAFKA_QUS/cooldown_strike_level_spark.py"
        )
    )

    start >> run_cooldown
