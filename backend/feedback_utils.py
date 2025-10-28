# backend/feedback_utils.py
import os, csv, datetime
from twilio.rest import Client
from dotenv import load_dotenv
import os
load_dotenv()




FEEDBACK_FILE = "data/processed/feedback_log.csv"

def send_feedback_request(user_phone, user_query, restaurant_list):
    """Send automated feedback message via WhatsApp."""
    client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    msg = (
        "Hi! Hope you enjoyed exploring restaurants today 😊\n\n"
        "Would you like to rate your experience (1–5 ⭐) or share feedback?\n"
        "Reply with:\n⭐ 1–5 for rating\n💬 Or any comment for suggestions."
    )
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

    print("Twilio SID:", account_sid)
    print("Twilio Token loaded:", auth_token)
    print("Twilio WhatsApp:", whatsapp_number)
    print("user_phone:", user_phone)
    client.messages.create(
        from_=os.getenv("TWILIO_WHATSAPP_NUMBER"),
        body=msg,
        to=f"whatsapp:{user_phone}"
    )
    print(f"📨 Feedback request sent to {user_phone}")

def save_feedback(user_phone, feedback_text):
    """Log user feedback."""
    os.makedirs("data/processed", exist_ok=True)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now(), user_phone, feedback_text])
    print(f"📝 Feedback saved: {feedback_text}")

def ensure_feedback_file():
    """Ensure feedback CSV exists."""
    if not os.path.exists(FEEDBACK_FILE):
        os.makedirs("data/processed", exist_ok=True)
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_phone", "feedback"])
