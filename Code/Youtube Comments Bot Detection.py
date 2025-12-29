from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import Counter
from json import JSONDecodeError
import time
import json
import pandas as pd
import os

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "output_combined.json")
CSV_PATH = os.path.join(BASE_DIR, "output_combined.csv")

# remove old video ids where data has already been collected 
# only include new video ids here
VIDEO_ID = ["HYYoepuLZSI", "zMNAos1hotI", "4ZzpL8BCKPw", "61Xlain6QP8", "5E1d_wA3B1g", "aprGWX2IOGs", "F-OU8gr2htQ", "VZFjtKM_GtE"]   

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found. Set YOUTUBE_API_KEY environment variable.")

youtube = build("youtube", "v3", developerKey=API_KEY)

# ---------------- HELPER FUNCTIONS ----------------

def get_processed_video_ids(existing_comments):
    """Extract video IDs that have already been processed."""
    return set(
        comment.get("videoID")
        for comment in existing_comments
        if "videoID" in comment
    )

def get_video_genre(video_id):
    """Fetch human-readable genre of a YouTube video."""
    try:
        video_resp = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        category_id = video_resp["items"][0]["snippet"]["categoryId"]
        cat_resp = youtube.videoCategories().list(
            part="snippet",
            id=category_id
        ).execute()
        return cat_resp["items"][0]["snippet"]["title"]
    except Exception as e:
        print(f"Error fetching genre for {video_id}: {e}")
        return None

def fetch_all_comment_threads(video_id, video_genre=None):
    """Fetch comments + channel info."""
    comments = []
    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    count = 0
    max_batches = 50
    comment_texts = []

    while request and count < max_batches:
        semicomments = []
        author_ids = set()

        try:
            resp = request.execute()
        except HttpError as e:
            print(f"HttpError while fetching comments: {e}")
            break

        for item in resp.get("items", []):
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

        request = youtube.commentThreads().list_next(request, resp)
        count += 1
        time.sleep(0.1)

    # Duplicate detection
    counter = Counter(comment_texts)
    for comment in comments:
        text = comment["commentText"]
        comment["isDuplicateComment"] = counter[text] > 1
        del comment["commentText"]

    return comments

# ---------------- MAIN ----------------

if __name__ == "__main__":

    # Load existing data if present
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                all_comments = json.load(f)
        except JSONDecodeError:
            print("Existing JSON corrupted. Resetting.")
            all_comments = []
    else:
        all_comments = []

    processed_video_ids = get_processed_video_ids(all_comments)

    print(f"Previously processed videos: {processed_video_ids}")

    for vid in VIDEO_ID:
        if vid in processed_video_ids:
            print(f"Skipping {vid} (already collected)")
            continue

        print(f"\n--- Fetching NEW video {vid} ---")
        genre = get_video_genre(vid)
        comments = fetch_all_comment_threads(vid, genre)
        all_comments.extend(comments)
        print(f"Added {len(comments)} comments")

    # Save updated data
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_comments, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(all_comments)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print("\nData collection complete.")
    print(f"Total comments stored: {len(df)}")
    print(df.head(10))
