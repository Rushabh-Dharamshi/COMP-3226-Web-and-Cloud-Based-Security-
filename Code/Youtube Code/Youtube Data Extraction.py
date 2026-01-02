from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import Counter
from json import JSONDecodeError
import time
import json
import pandas as pd
import os
import matplotlib.pyplot as plt

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

def get_video_ids_from_excel(excel_path):
    """Load video IDs from Excel file."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"{excel_path} not found. Create Excel with column 'videoID'.")
    df = pd.read_excel(excel_path, engine='openpyxl')
    return df['videoID'].dropna().astype(str).tolist()

def get_processed_video_ids(existing_comments):
    """Return set of already processed video IDs."""
    return set(comment.get("videoID") for comment in existing_comments if "videoID" in comment)

def get_video_genre(video_id):
    """Fetch human-readable genre of a YouTube video."""
    try:
        video_resp = youtube.videos().list(part="snippet", id=video_id).execute()
        category_id = video_resp["items"][0]["snippet"]["categoryId"]
        cat_resp = youtube.videoCategories().list(part="snippet", id=category_id).execute()
        return cat_resp["items"][0]["snippet"]["title"]
    except Exception as e:
        print(f"Error fetching genre for {video_id}: {e}")
        return "Unknown"


def fetch_all_comment_threads(video_id, video_genre=None, max_comments=5000):
    """Fetch comments + channel info for a video, up to max_comments."""
    comments = []
    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    comment_texts = []
    total_fetched = 0  # Track number of comments fetched

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
                break  # Stop if reached limit
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
            break  # Stop fetching more pages
        request = youtube.commentThreads().list_next(request, resp)
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
    # Load Excel video IDs
    video_ids_raw = get_video_ids_from_excel(EXCEL_PATH)
    total_ids = len(video_ids_raw)
    print(f"{total_ids} total video IDs loaded from Excel.")

    # Detect duplicates in Excel
    duplicates_in_excel = [vid for vid, count in Counter(video_ids_raw).items() if count > 1]
    if duplicates_in_excel:
        print(f"Warning: {len(duplicates_in_excel)} duplicate video IDs found in Excel.")
        # Optional: print duplicates
        print("Duplicate video IDs:", duplicates_in_excel)

    # Keep only unique IDs from Excel for processing
    unique_video_ids = list(dict.fromkeys(video_ids_raw))
    print(f"{len(unique_video_ids)} unique video IDs will be processed.")

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

    # Set of already processed video IDs (from previous runs or JSON)
    processed_video_ids = get_processed_video_ids(all_comments)
    print(f"Previously processed videos: {len(processed_video_ids)}")

    # Process videos in order, skipping duplicates in Excel and JSON
    for vid in unique_video_ids:
        if vid in processed_video_ids:
            print(f"Skipping {vid} (already collected)")
            continue

        print(f"\n--- Fetching NEW video {vid} ---")
        genre = get_video_genre(vid)
        comments = fetch_all_comment_threads(vid, genre)
        all_comments.extend(comments)
        print(f"Added {len(comments)} comments from {vid}")

        # Mark this video as processed immediately to skip any duplicates
        processed_video_ids.add(vid)

        # Save JSON incrementally (with line terminator normalization)
        json_text = json.dumps(all_comments, indent=2, ensure_ascii=False)
        json_text = json_text.replace("\u2028", "\n").replace("\u2029", "\n")
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            f.write(json_text)

    # ---------------- SAVE CSV ----------------
    df = pd.DataFrame(all_comments)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    # ---------------- SUMMARY ----------------
    print("\nData collection complete.")
    print(f"Total comments stored: {len(df)}")
    print(f"Total unique videos processed (including previous runs): {len(processed_video_ids)}")

    summary_df = df[['videoID','videoGenre']].drop_duplicates()
    summary_counts = summary_df['videoGenre'].value_counts()
    print("\nSummary of videos per category:")
    print(summary_counts)

    # ---------------- PIE CHART ----------------
    plt.figure(figsize=(8,8))
    summary_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='tab20')
    plt.title("Distribution of Videos by Category")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
