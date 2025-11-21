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

API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_ID = "NQypHE9_Fm4"  # change as needed

if not API_KEY:
    raise ValueError("API_KEY not found. Make sure YOUTUBE_API_KEY is set.")

youtube = build('youtube', 'v3', developerKey=API_KEY)

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
        genre = cat_resp["items"][0]["snippet"]["title"]
        return genre
    except Exception as e:
        print(f"Error fetching video genre: {e}")
        return None

def fetch_all_comment_threads(video_id, video_genre=None):
    """Fetch all comments from a video, including channel info."""
    comments = []
    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,        
        textFormat="plainText"
    )
    count = 0
    max_batches = 50  # change to control max comments

    while request and count < max_batches:
        commentAuthorIdSet = set()
        semicomments = []
        try:
            resp = request.execute()
        except HttpError as e:
            print(f"HttpError: {e}. Ending process.")
            return comments
        
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            author_id = top.get("authorChannelId", {}).get("value")
            print(f"Found user: {top.get('authorDisplayName')}")
            comment = {
                "authorChannelId": author_id,
                "commentText": top.get("textDisplay"),
                "publishedAt": top.get("publishedAt"),
                "commentLikeCount": top.get("likeCount"),
            }
            semicomments.append(comment)
            if author_id:
                commentAuthorIdSet.add(author_id)
        
        # Fetch channel info in batches
        channelIdDict = {}
        commentAuthorIdList = list(commentAuthorIdSet)
        for i in range(0, len(commentAuthorIdList), 50):
            batch = commentAuthorIdList[i:i+50]
            try:
                channelresp = youtube.channels().list(
                    part="snippet,statistics",
                    id=",".join(batch),
                    maxResults=50
                ).execute()
            except HttpError as e:
                print(f"HttpError: {e}. Ending process.")
                return comments
            for item in channelresp.get("items", []):
                channelIdDict[item["id"]] = {
                    "title": item["snippet"]["title"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "viewCount": item["statistics"].get("viewCount"),
                    "subscriberCount": item["statistics"].get("subscriberCount"),
                    "videoCount": item["statistics"].get("videoCount"),
                    "country": item["snippet"].get("country"),
                }
        
        # Combine comment + channel info
        for item in semicomments:
            channelDict = channelIdDict.get(item["authorChannelId"], {})
            comments.append({
                "channelTitle": channelDict.get("title"),
                "channelDate": channelDict.get("publishedAt"),
                "channelViewCount": channelDict.get("viewCount"),
                "channelSubscriberCount": channelDict.get("subscriberCount"),
                "channelVideoCount": channelDict.get("videoCount"),
                "channelCountry": channelDict.get("country"),
                "channelID": item["authorChannelId"],
                "commentText": item["commentText"],
                "commentDate": item["publishedAt"],
                "commentLikeCount": item["commentLikeCount"],
                "videoGenre": video_genre
            })
        
        request = youtube.commentThreads().list_next(request, resp)
        count += 1
        time.sleep(0.1)
    
    return comments

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load existing comments safely
    if not os.path.exists(JSON_PATH):
        with open(JSON_PATH, "w") as f:
            json.dump([], f)

    try:
        with open(JSON_PATH, "r") as f:
            old_comments = json.load(f)
    except JSONDecodeError:
        print("Existing JSON corrupted, resetting file.")
        old_comments = []
        with open(JSON_PATH, "w") as f:
            json.dump([], f)
    
    # Get video genre
    video_genre = get_video_genre(VIDEO_ID)

    # Fetch comments
    comments = fetch_all_comment_threads(VIDEO_ID, video_genre)

    # Save combined JSON
    all_comments = old_comments + comments
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_comments, f, indent=2, ensure_ascii=False)

    # Save CSV
    df = pd.DataFrame(all_comments)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} comments to {CSV_PATH}")
    print(df.head(10))

    youtube.close()
