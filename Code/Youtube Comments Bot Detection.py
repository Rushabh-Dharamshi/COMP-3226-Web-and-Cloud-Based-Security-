from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from json import JSONDecodeError
import time
import json
import pandas as pd
import os

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "output_combined.json")
CSV_PATH = os.path.join(BASE_DIR, "output_combined.csv")

# IMPORTANT:
# You can swap the VIDEO_ID here. Both files had different IDs.
API_KEY = "AIzaSyAh3tkglLdIpG7zmTI2Gu500SN8nw4eNs8" #replace with your own API key
VIDEO_IDS = [
    "SdSSPF1S-Uc",
    "t6h8Uae2Q_E",
    "mfv0V1SxbNA",
    "KipDBa4bTl8",
    "tAXKxKTGWFQ",
    "lrh1KcM2QWo",
    "uXD6x3pfYxQ",
    "hmdzniMJOZs",
    "yFmizY7pe7U",
    "OPZoLZBXD0Y",
    "4xnqtbrLBBQ",
    "GX8Hg6kWQYI",
    "C1f8_-8GU7I",
    "gqwkG4W5Xm0",
    "uTWtdwzzxaM",
    "F9EuA8mtHV8",
    "RGoQCy2BlMw",
    "KGBGXZTQ_KI",
    "nkK312fMPsQ",
    "PCpvt3wk42A",
    "3olqrQtjPfc",
    "7SopjMoIMfg",
    "lz3088bex3w",
    "COtZc4S4rdY",
    "K_o2ejRHLws",
    "7DKv5H5Frt0",
    "RbXzz3oZDr4",
    "NBZv0_MImIY",
    "PdmoGCjHpE0",
    "jlV9kJDdYug"
]

if not API_KEY:
    raise ValueError("API_KEY not found. Make sure YOUTUBE_API_KEY is set.")

youtube = build("youtube", "v3", developerKey=API_KEY)


# ---------------- FUNCTIONS ----------------

def get_video_genre(video_id):
    """Fetch the human-readable genre of a YouTube video."""
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
        print(f"Error fetching video genre: {e}")
        return None


def fetch_all_comment_threads(video_id, video_genre=None):
    """Fetch comments + channel info + extended features."""

    comments = []

    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    count = 0
    max_batches = 50  # limit for throttling

    while request and count < max_batches:
        semicomments = []
        author_ids = set()

        try:
            resp = request.execute()
        except HttpError as e:
            print(f"HttpError: {e}. Ending comment fetch.")
            return comments

        # Extract comment data
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            author_id = top.get("authorChannelId", {}).get("value")

            print("Found user:", top.get("authorDisplayName"))

            semicomments.append({
                "authorChannelId": author_id,
                "commentText": top.get("textDisplay"),
                "publishedAt": top.get("publishedAt"),
                "commentLikeCount": top.get("likeCount"),
            })

            if author_id:
                author_ids.add(author_id)

        # Fetch channel info (batched)
        channel_data = {}
        author_list = list(author_ids)

        for i in range(0, len(author_list), 50):
            batch = author_list[i:i + 50]

            try:
                ch_resp = youtube.channels().list(
                    part="snippet,statistics",
                    id=",".join(batch),
                    maxResults=50
                ).execute()
            except HttpError:
                print("HttpError fetching channel data. Ending.")
                return comments

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

                    # EXTRA FEATURES FROM YOUTUBE.PY
                    "hasDescription": len(description.strip()) > 0,
                    "channelDescription": description, 
                    "defaultProfilePic": ("default" in thumb_url or "channels/default" in thumb_url)
                }

        # Combine comment + channel info
        for item in semicomments:
            ch = channel_data.get(item["authorChannelId"], {})

            comments.append({
                "videoId": video_id,
                "channelTitle": ch.get("title"),
                "channelDate": ch.get("publishedAt"),
                "channelViewCount": ch.get("viewCount"),
                "channelSubscriberCount": ch.get("subscriberCount"),
                "channelVideoCount": ch.get("videoCount"),
                "channelCountry": ch.get("country"),
                "channelID": item["authorChannelId"],

                "commentText": item["commentText"],
                "commentDate": item["publishedAt"],
                "commentLikeCount": item["commentLikeCount"],

                "videoGenre": video_genre,

                # EXTRA FIELDS (safe even if channel missing)
                "hasDescription": ch.get("hasDescription"),
                "channelDescription": ch.get("channelDescription"), 
                "defaultProfilePic": ch.get("defaultProfilePic")
            })

        request = youtube.commentThreads().list_next(request, resp)
        count += 1
        time.sleep(0.1)

    return comments


# ---------------- MAIN ----------------
if __name__ == "__main__":

    # Ensure JSON file exists
    if not os.path.exists(JSON_PATH):
        with open(JSON_PATH, "w") as f:
            json.dump([], f)

    # Load old comments safely
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            old_comments = json.load(f)
    except JSONDecodeError:
        print("JSON corrupted. Resetting.")
        old_comments = []
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Fetch video metadata
    all_new_comments = []

    for video_id in VIDEO_IDS:
        print(f"\nFetching data for video: {video_id}")

        video_genre = get_video_genre(video_id)
        comments = fetch_all_comment_threads(video_id, video_genre)

        print(f"Collected {len(comments)} comments from {video_id}")
        all_new_comments.extend(comments)

    # Merge & save JSON
    all_comments = old_comments + all_new_comments
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_comments, f, indent=2, ensure_ascii=False)

    # Save CSV (feature from test.py)
    df = pd.DataFrame(all_comments)
    df.to_csv(CSV_PATH, index=False)

    print(f"Saved {len(df)} comments → {CSV_PATH}")
    print(df.head(10))

    youtube.close()