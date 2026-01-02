import os
import json
import pandas as pd
import re

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Youtube Data")

JSON_PATH = os.path.join(DATA_DIR, "Youtube_extracted_data.json")
CSV_PATH = os.path.join(DATA_DIR, "Youtube_extracted_data.csv")  # Same name as JSON, just .csv

# ---------------- LOAD AND CLEAN JSON ----------------
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"{JSON_PATH} not found. Cannot create CSV backup.")

def clean_json_text(text):
    """Remove invalid control characters from JSON."""
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_text = f.read()

json_text_clean = clean_json_text(json_text)

try:
    all_comments = json.loads(json_text_clean)
except json.JSONDecodeError as e:
    print("JSON still invalid after cleaning:", e)
    raise

# ---------------- CREATE CSV ----------------
if not all_comments:
    print("⚠ No comments found in JSON. CSV will not be created.")
else:
    df = pd.DataFrame(all_comments)

    # Save to CSV
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ CSV backup created from JSON at: {CSV_PATH}")
    print(f"Total comments exported: {len(df)}")
