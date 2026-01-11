"""
Check if usage data is being logged to MongoDB
"""
from pymongo import MongoClient
from config import CONFIG
from datetime import datetime, timedelta

# Connect to MongoDB
client = MongoClient(CONFIG["mongo_uri"])
db = client[CONFIG["mongo_db"]]

print("=" * 60)
print("📊 Checking Usage Data in MongoDB")
print("=" * 60)

# Check usage_logs collection
usage_logs = db.usage_logs
total_logs = usage_logs.count_documents({})
print(f"\n📝 Total usage logs: {total_logs}")

if total_logs > 0:
    print("\n🔍 Recent logs (last 5):")
    recent_logs = usage_logs.find().sort("timestamp", -1).limit(5)
    for log in recent_logs:
        print(f"  - {log.get('timestamp')} | User: {log.get('user_id')} | Type: {log.get('operation_type')} | Endpoint: {log.get('endpoint')}")

# Check daily_stats collection
daily_stats = db.daily_stats
total_stats = daily_stats.count_documents({})
print(f"\n📈 Total daily stats: {total_stats}")

if total_stats > 0:
    print("\n🔍 Recent stats:")
    recent_stats = daily_stats.find().sort("date", -1).limit(3)
    for stat in recent_stats:
        print(f"  - {stat.get('date')} | Ops: {stat.get('total_operations')} | Tokens: {stat.get('total_tokens')}")

# Check for logs in the last hour
one_hour_ago = datetime.utcnow() - timedelta(hours=1)
recent_count = usage_logs.count_documents({"timestamp": {"$gte": one_hour_ago}})
print(f"\n⏰ Logs in last hour: {recent_count}")

# Check unique users
pipeline = [
    {"$group": {"_id": "$user_id"}},
    {"$count": "unique_users"}
]
result = list(usage_logs.aggregate(pipeline))
unique_users = result[0]["unique_users"] if result else 0
print(f"\n👥 Unique users with activity: {unique_users}")

print("\n" + "=" * 60)

client.close()
