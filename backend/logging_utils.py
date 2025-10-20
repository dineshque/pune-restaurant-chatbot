# backend/logging_utils.py
import csv, os
from datetime import datetime

LOG_FILE = "data/processed/query_logs.csv"
os.makedirs("data/processed", exist_ok=True)

def log_query(query:str, results:list, latency:float, extras:dict|None=None):
    headers = ["timestamp","query","latency_sec","top_names","extras"]
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "latency_sec": round(latency, 3),
        "top_names": " | ".join([r["name"] for r in results[:5]]),
        "extras": extras or {}
    }
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if write_header: w.writeheader()
        w.writerow(row)
