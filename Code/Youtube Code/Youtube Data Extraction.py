from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import Counter
import json
import pandas as pd
import os
import time
import re

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "Youtube Data")
os.makedirs(DATA_DIR, exist_ok=True)

JSON_PATH = os.path.join(DATA_DIR, "Youtube_extracted_data.json")
CSV_PATH = os.path.join(DATA_DIR, "Youtube_extracted_data.csv")
EXCEL_PATH = os.path.join(DATA_DIR, "video_ids.xlsx")

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found. Set YOUTUBE_API_KEY environment variable.")

youtube = build("youtube", "v3", developerKey=API_KEY)

# ---------------- HELPER FUNCTIONS ----------------

def clean_and_load_json(json_path):
    """Load a JSON file and clean it in place if corrupted."""
    if not os.path.exists(json_path):
        print(f"⚠ {json_path} does not exist. Starting with empty list.")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"✅ Loaded {len(data)} records from {json_path}")
            return data
    except json.JSONDecodeError:
        print(f"⚠ {json_path} is corrupted. Cleaning in place...")
        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # Extract objects containing "videoID"
        matches = re.findall(r'{[^{}]*"videoID"\s*:\s*"[^"]+"[^{}]*}', text)
        recovered = []
        for m in matches:
            try:
                recovered.append(json.loads(m))
            except json.JSONDecodeError:
                continue
        # Save cleaned JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(recovered, f, indent=2, ensure_ascii=False)
        print(f"✅ Cleaned {len(recovered)} records in {json_path}")
        return recovered

def get_video_ids_from_excel(excel_path):
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"{excel_path} not found. Create Excel with column 'videoID'.")
    df = pd.read_excel(excel_path, engine='openpyxl')
    return df['videoID'].dropna().astype(str).tolist()

def get_video_genre(video_id):
    try:
        video_resp = youtube.videos().list(part="snippet", id=video_id).execute()
        category_id = video_resp["items"][0]["snippet"]["categoryId"]
        cat_resp = youtube.videoCategories().list(part="snippet", id=category_id).execute()
        return cat_resp["items"][0]["snippet"]["title"]
    except Exception as e:
        print(f"Error fetching genre for {video_id}: {e}")
        return "Unknown"

def fetch_all_comment_threads(video_id, video_genre=None, max_comments=5000):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    comment_texts = []
    total_fetched = 0

    while request and total_fetched < max_comments:
        semicomments = []
        author_ids = set()
        try:
            resp = request.execute()
        except HttpError as e:
            print(f"HttpError while fetching comments for {video_id}: {e}")
            break

        for item in resp.get("items", []):
            if total_fetched >= max_comments:
                break
            top = item["snippet"]["topLevelComment"]["snippet"]
            author_id = top.get("authorChannelId", {}).get("value")
            semicomments.append({
                "authorChannelId": author_id,
                "commentText": top.get("textDisplay"),
                "publishedAt": top.get("publishedAt"),
                "commentLikeCount": top.get("likeCount"),
            })
            if author_id:
                author_ids.add(author_id)
            total_fetched += 1

        # Fetch channel metadata
        channel_data = {}
        author_list = list(author_ids)
        for i in range(0, len(author_list), 50):
            try:
                ch_resp = youtube.channels().list(
                    part="snippet,statistics",
                    id=",".join(author_list[i:i + 50])
                ).execute()
            except HttpError:
                continue
            for ch in ch_resp.get("items", []):
                snippet = ch["snippet"]
                stats = ch["statistics"]
                thumb_url = snippet["thumbnails"]["default"]["url"]
                description = snippet.get("description", "")
                channel_data[ch["id"]] = {
                    "title": snippet["title"],
                    "publishedAt": snippet["publishedAt"],
                    "viewCount": stats.get("viewCount"),
                    "subscriberCount": stats.get("subscriberCount"),
                    "videoCount": stats.get("videoCount"),
                    "country": snippet.get("country"),
                    "hasDescription": len(description.strip()) > 0,
                    "channelDescription": description,
                    "defaultProfilePic": ("default" in thumb_url or "channels/default" in thumb_url)
                }

        for item in semicomments:
            ch = channel_data.get(item["authorChannelId"], {})
            comment_texts.append(item["commentText"])
            comments.append({
                "videoID": video_id,
                "videoGenre": video_genre,
                "channelID": item["authorChannelId"],
                "channelTitle": ch.get("title"),
                "channelDate": ch.get("publishedAt"),
                "channelViewCount": ch.get("viewCount"),
                "channelSubscriberCount": ch.get("subscriberCount"),
                "channelVideoCount": ch.get("videoCount"),
                "channelCountry": ch.get("country"),
                "commentDate": item["publishedAt"],
                "commentLikeCount": item["commentLikeCount"],
                "commentText": item["commentText"],
                "hasDescription": ch.get("hasDescription"),
                "channelDescription": ch.get("channelDescription"),
                "defaultProfilePic": ch.get("defaultProfilePic")
            })

        if total_fetched >= max_comments:
            break
        request = youtube.commentThreads().list_next(request, resp)
        time.sleep(0.1)

    # Duplicate detection
    counter = Counter(comment_texts)
    for comment in comments:
        text = comment["commentText"]
        comment["isDuplicateComment"] = counter[text] > 1
        del comment["commentText"]

    return comments

def recover_video_ids_from_json_data(json_data):
    """Return set of videoIDs from loaded JSON data."""
    return set(c.get("videoID") for c in json_data if "videoID" in c)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    print("🔹 Starting YouTube data extraction...")

    # Clean and load main JSON
    original_comments = clean_and_load_json(JSON_PATH)
    processed_video_ids = recover_video_ids_from_json_data(original_comments)
    print(f"{len(original_comments)} comments already in main JSON.")
    print(f"{len(processed_video_ids)} videos already processed.\n")

    # Load video IDs from Excel
    video_ids_raw = get_video_ids_from_excel(EXCEL_PATH)
    unique_video_ids = list(dict.fromkeys(video_ids_raw))
    print(f"{len(unique_video_ids)} unique video IDs to process.\n")

    all_new_comments = []

    for vid in unique_video_ids:
        if vid in processed_video_ids:
            print(f"Skipping {vid} (already processed).")
            continue

        print(f"\n--- Fetching NEW video {vid} ---")
        genre = get_video_genre(vid)
        comments = fetch_all_comment_threads(vid, video_genre=genre)
        print(f"Fetched {len(comments)} comments from {vid}")

        # Avoid duplicates before merging
        existing_keys = set((c.get("videoID"), c.get("commentDate"), c.get("channelID")) for c in original_comments)
        new_comments = [c for c in comments if (c.get("videoID"), c.get("commentDate"), c.get("channelID")) not in existing_keys]

        original_comments.extend(new_comments)
        all_new_comments.extend(new_comments)
        processed_video_ids.add(vid)
        print(f"✅ {len(new_comments)} new comments merged into main JSON.\n")

        # Save main JSON incrementally to avoid data loss
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(original_comments, f, indent=2, ensure_ascii=False)

    # Save CSV
    if all_new_comments:
        df = pd.DataFrame(all_new_comments)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ CSV saved: {CSV_PATH}")

    print(f"\n🔹 Data extraction complete. Only 2 files remain: main JSON and CSV.")
    print(f"Total comments in JSON: {len(original_comments)}")
    print(f"Total new comments in CSV: {len(all_new_comments)}")
