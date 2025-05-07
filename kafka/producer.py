from kafka import KafkaProducer
import json
import time

# Load messages
with open("/home/ubuntu/kafka_codes/KAFKA_QUS/messages.json", "r") as f:
    messages = json.load(f)

# Set up Kafka producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Send messages one by one (simulate real-time)
for msg in messages:
    producer.send("test-topic", msg)
    print(f"✅ Sent: {msg}")
    time.sleep(2)  # simulate delay

producer.flush()
producer.close()
