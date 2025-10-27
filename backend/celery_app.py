# backend/celery_app.py
from celery import Celery
import os

# ---------------------------------------------------------------------
# Redis Configuration
# ---------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "restaurant_feedback",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.feedback_tasks"]  # 👈 auto-import feedback tasks
)

# ---------------------------------------------------------------------
# Celery Configuration
# ---------------------------------------------------------------------
celery_app.conf.update(
    timezone="Asia/Kolkata",
    enable_utc=False,
    broker_connection_retry_on_startup=True,  # 👈 suppress startup warnings
)

# Optional: periodic tasks can be added later here if needed
# celery_app.conf.beat_schedule = {
#     "send-feedback-every-3-hours": {
#         "task": "backend.feedback_tasks.send_feedback_reminders",
#         "schedule": 3 * 60 * 60,  # every 3 hours
#     },
# }

# ---------------------------------------------------------------------
# Import tasks explicitly for safety
# ---------------------------------------------------------------------
import backend.feedback_tasks  # 👈 ensures task registration

print("✅ Celery initialized with broker:", REDIS_URL)
