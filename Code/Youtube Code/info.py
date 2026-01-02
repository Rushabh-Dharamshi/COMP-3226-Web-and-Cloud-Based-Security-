import os
import json
import pandas as pd
import re

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "Youtube Data")
JSON_PATH = os.path.join(DATA_DIR, "Youtube_extracted_data.json")

# ---------------- LOAD DATA (robust) ----------------
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"{JSON_PATH} not found. Make sure the JSON data exists.")

def clean_json_text(text):
    """Remove invalid control characters."""
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

# Read JSON file
with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_text = f.read()

# Clean invalid characters
json_text_clean = clean_json_text(json_text)

# Load JSON
try:
    all_comments = json.loads(json_text_clean)
except json.JSONDecodeError as e:
    print("Still invalid JSON after cleaning:", e)
    raise

# ---------------- CREATE DATAFRAME ----------------
df = pd.DataFrame(all_comments)

# Ensure required columns exist
if 'videoID' not in df.columns or 'videoGenre' not in df.columns:
    raise ValueError("JSON must contain 'videoID' and 'videoGenre' fields.")

# Keep only unique videoID per category
unique_videos = df[['videoID', 'videoGenre']].drop_duplicates()

# ---------------- GROUP BY CATEGORY ----------------
videos_grouped = unique_videos.groupby('videoGenre')['videoID'].apply(list)

# ---------------- PRINT SUMMARY ----------------
total_videos = unique_videos['videoID'].nunique()
print(f"Total unique videos extracted: {total_videos}\n")

print("Videos per category:\n")
for category, vids in videos_grouped.items():
    print(f"Category: {category} -> {len(vids)} videos")
