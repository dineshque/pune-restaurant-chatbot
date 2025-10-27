import pandas as pd, time, datetime
from backend.feedback_utils import send_feedback_request
from backend.logging_utils import LOG_FILE  # path to query logs

def get_recent_users(hours=3):
    """Fetch unique users who queried recently."""
    df = pd.read_csv(LOG_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=hours)
    recent = df[df["timestamp"] > cutoff]
    return recent["user_phone"].unique().tolist()

def send_pending_feedbacks():
    """Send feedback requests for recent users."""
    users = get_recent_users()
    for u in users:
        send_feedback_request(u, "", [])
    print(f"✅ Sent feedback requests to {len(users)} users at {datetime.datetime.now()}")

if __name__ == "__main__":
    send_pending_feedbacks()
