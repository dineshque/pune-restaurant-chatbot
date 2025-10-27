from backend.celery_app import celery_app
from backend.feedback_utils import send_feedback_request
import datetime

@celery_app.task(name="backend.feedback_tasks.delayed_feedback")
def delayed_feedback(user_phone, query, restaurants):
    """Send feedback request asynchronously after delay."""
    print(f"[{datetime.datetime.now()}] Sending delayed feedback to {user_phone}...")
    send_feedback_request(user_phone, query, restaurants)

def schedule_feedback_request(user_phone, query, restaurants, delay_sec=20):
    """Schedule feedback reminder 1 minute after conversation end."""
    if not user_phone:
        print("⚠️ No user phone — skipping feedback scheduling.")
        return
    delayed_feedback.apply_async((user_phone, query, restaurants), countdown=delay_sec)
    print(f"⏳ Feedback reminder scheduled for {user_phone} in {delay_sec} seconds.")
