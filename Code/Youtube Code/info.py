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
required_cols = {'videoID', 'videoGenre'}
if not required_cols.issubset(df.columns):
    raise ValueError("JSON must contain 'videoID' and 'videoGenre' fields.")

# ---------------- OVERALL COUNTS ----------------
total_comments = len(df)
total_unique_videos = df['videoID'].nunique()

print("=" * 50)
print("OVERALL DATASET STATISTICS")
print("=" * 50)
print(f"Total comments extracted: {total_comments}")
print(f"Total unique videos extracted: {total_unique_videos}\n")

# ---------------- VIDEOS PER CATEGORY ----------------
videos_per_category = (
    df[['videoID', 'videoGenre']]
    .drop_duplicates()
    .groupby('videoGenre')
    .size()
    .sort_values(ascending=False)
)

print("=" * 50)
print("VIDEOS PER CATEGORY")
print("=" * 50)
for category, count in videos_per_category.items():
    print(f"{category}: {count} videos")

# ---------------- COMMENTS PER CATEGORY ----------------
comments_per_category = (
    df
    .groupby('videoGenre')
    .size()
    .sort_values(ascending=False)
)

print("\n" + "=" * 50)
print("COMMENTS PER CATEGORY")
print("=" * 50)
for category, count in comments_per_category.items():
    print(f"{category}: {count} comments")
