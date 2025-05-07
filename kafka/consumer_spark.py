from kafka import KafkaConsumer, TopicPartition
import json
import os
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StringType, DateType, IntegerType

# ✅ Configs
CHECKPOINT_FILE = "/home/ubuntu/kafka_codes/KAFKA_QUS/checkpoint/kafka_checkpoint.json"
TOPIC_NAME = "test-topic"
BOOTSTRAP_SERVERS = "localhost:9092"

# ✅ JDBC JAR path
jdbc_jar_path = "/home/ubuntu/postgresql-42.7.5.jar"
assert os.path.exists(jdbc_jar_path), f"❌ JDBC JAR not found at {jdbc_jar_path}"

# ✅ Spark session
spark = SparkSession.builder \
    .appName("KafkaCustomCheckpointConsumer") \
    .config("spark.jars", jdbc_jar_path) \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ✅ Load bad words
with open("/home/ubuntu/kafka_codes/KAFKA_QUS/marked_word.json", "r") as f:
    bad_words = set(word.strip().lower() for word in json.load(f))

# ✅ PostgreSQL config
pg_url = "jdbc:postgresql://3.221.182.234:5432/test_topic"
pg_table = "flagged_messages"
pg_user = "test_user"
pg_pass = "test_user"

# ✅ Data schema
schema = StructType() \
    .add("sender_id", StringType()) \
    .add("receiver_id", StringType()) \
    .add("message", StringType()) \
    .add("date", DateType()) \
    .add("strike_count", IntegerType())

# ✅ Utility: count flagged words
def count_bad_words(message):
    words = message.lower().split()
    return 1 if any(word in bad_words for word in words) else 0

# ✅ Load checkpoint from file
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
            return {
                TopicPartition(tp["topic"], tp["partition"]): tp["offset"]
                for tp in data
            }
    return {}

# ✅ Save checkpoint to file
def save_checkpoint(offsets):
    data = [
        {"topic": tp.topic, "partition": tp.partition, "offset": offset}
        for tp, offset in offsets.items()
    ]
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

# ✅ Safe deserializer
def safe_json_deserializer(m):
    try:
        return json.loads(m.decode("utf-8"))
    except Exception as e:
        print(f"❌ Deserialization error: {e}")
        return None

# ✅ Kafka consumer (no group ID, manual assign/seek)
consumer = KafkaConsumer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    value_deserializer=safe_json_deserializer
)

# ✅ Assign partitions and seek to last checkpoint
checkpoint = load_checkpoint()
partitions = [TopicPartition(TOPIC_NAME, p) for p in consumer.partitions_for_topic(TOPIC_NAME)]
consumer.assign(partitions)

for tp in partitions:
    if tp in checkpoint:
        print(f"⏪ Resuming {tp} from offset {checkpoint[tp]}")
        consumer.seek(tp, checkpoint[tp])
    else:
        print(f"📍 No checkpoint for {tp}, consuming from default")

# ✅ Buffering logic
buffer = []
flush_interval = timedelta(seconds=5)
flush_max_messages = 10
last_flush_time = datetime.now()

print("🚀 Kafka consumer started...")

# ✅ Main loop with full safety
for msg in consumer:
    try:
        data = msg.value
        if not isinstance(data, dict):
            print(f"⚠️ Skipping invalid message (not a dict): {data}")
            continue

        sender = data.get("sender", "")
        receiver = data.get("receiver", "")
        message = data.get("message", "")

        strike_count = count_bad_words(message)

        if strike_count > 0:
            print(f"🚨 FLAGGED: '{message}' by {sender} ({strike_count} strike)")

            row = Row(
                sender_id=sender,
                receiver_id=receiver,
                message=message,
                date=datetime.now().date(),
                strike_count=strike_count
            )
            buffer.append((msg, row))

    except Exception as e:
        print(f"❌ Error processing message: {e}")
        continue

    # ✅ Flush if needed
    if datetime.now() - last_flush_time >= flush_interval or len(buffer) >= flush_max_messages:
        if buffer:
            print(f"📤 Flushing {len(buffer)} flagged messages to Postgres...")

            rows = [r for (_, r) in buffer]
            df = spark.createDataFrame(rows, schema=schema)
            df.show(truncate=False)

            try:
                df.write \
                    .format("jdbc") \
                    .option("url", pg_url) \
                    .option("driver", "org.postgresql.Driver") \
                    .option("dbtable", pg_table) \
                    .option("user", pg_user) \
                    .option("password", pg_pass) \
                    .mode("append") \
                    .save()

                # ✅ Track latest offsets
                latest_offsets = {}
                for m, _ in buffer:
                    tp = TopicPartition(m.topic, m.partition)
                    latest_offsets[tp] = m.offset + 1

                save_checkpoint(latest_offsets)
                print("✅ Offsets committed to file.")

            except Exception as e:
                print(f"❌ Error writing to PostgreSQL: {e}")
                print("⚠️ Retaining buffer, not committing offsets.")

            buffer.clear()
            last_flush_time = datetime.now()




# from kafka import KafkaConsumer
# import json
# import os
# from datetime import datetime, timedelta
# from pyspark.sql import SparkSession, Row
# from pyspark.sql.types import StructType, StringType, DateType, IntegerType

# # ✅ JDBC JAR path
# jdbc_jar_path = "/home/himanshu/postgresql-42.7.5.jar"
# assert os.path.exists(jdbc_jar_path), f"❌ JDBC JAR not found at {jdbc_jar_path}"

# # ✅ Start Spark Session with JDBC driver
# spark = SparkSession.builder \
#     .appName("KafkaFlaggedMessageConsumer") \
#     .config("spark.jars", jdbc_jar_path) \
#     .getOrCreate()

# spark.sparkContext.setLogLevel("WARN")

# # ✅ Load bad words
# marked_words_path = "/home/himanshu/Learning/bootcamp_project/KAFKA_QUS/marked_word.json"
# with open(marked_words_path, "r") as f:
#     bad_words = set(word.strip().lower() for word in json.load(f))

# # ✅ Kafka Consumer Setup
# def safe_json_deserializer(m):
#     try:
#         return json.loads(m.decode("utf-8"))
#     except Exception as e:
#         print(f"❌ JSON parse error: {e}")
#         return None

# consumer = KafkaConsumer(
#     "employee_messages",
#     bootstrap_servers="localhost:9092",
#     auto_offset_reset="latest",
#     enable_auto_commit=True,
#     value_deserializer=safe_json_deserializer
# )

# # ✅ Schema definition
# schema = StructType() \
#     .add("sender_id", StringType()) \
#     .add("receiver_id", StringType()) \
#     .add("message", StringType()) \
#     .add("date", DateType()) \
#     .add("strike_count", IntegerType())

# # ✅ Bad word counter
# # def count_bad_words(message):
# #     words = message.lower().split()
# #     return sum(word in bad_words for word in words)

# def count_bad_words(message):
#     words = message.lower().split()
#     return 1 if any(word in bad_words for word in words) else 0


# # ✅ PostgreSQL config
# pg_url = "jdbc:postgresql://localhost:5432/test_topic"
# pg_table = "flagged_messages"
# pg_user = "postgres"
# pg_pass = "postgres"

# # ✅ Buffer and timer
# buffer = []
# flush_interval = timedelta(seconds=7)
# last_flush_time = datetime.now()

# print("🚀 Kafka consumer started...")

# # ✅ Main consumer loop
# for msg in consumer:
#     data = msg.value
#     if not data:
#         continue

#     sender = data.get("sender", "")
#     receiver = data.get("receiver", "")
#     message = data.get("message", "")

#     strike_count = count_bad_words(message)

#     print(f"🔔 Message from {sender} | Strikes: {strike_count}")

#     if strike_count > 0:
#         print(f"🚨 FLAGGED: '{message}' by {sender} ({strike_count} strikes)")

#         row = Row(
#             sender_id=sender,
#             receiver_id=receiver,
#             message=message,
#             date=datetime.now().date(),
#             strike_count=strike_count
#         )
#         buffer.append(row)

#     # ✅ Flush every X seconds
#     if datetime.now() - last_flush_time >= flush_interval and buffer:
#         print(f"📤 Writing {len(buffer)} flagged messages to Postgres...")

#         df = spark.createDataFrame(buffer, schema=schema)
#         df.show(truncate=False)

#         df.write \
#             .format("jdbc") \
#             .option("url", pg_url) \
#             .option("driver", "org.postgresql.Driver") \
#             .option("dbtable", pg_table) \
#             .option("user", pg_user) \
#             .option("password", pg_pass) \
#             .mode("append") \
#             .save()

#         buffer.clear()
#         last_flush_time = datetime.now()
#         print(f"✅ Batch saved at {last_flush_time.strftime('%H:%M:%S')}")
