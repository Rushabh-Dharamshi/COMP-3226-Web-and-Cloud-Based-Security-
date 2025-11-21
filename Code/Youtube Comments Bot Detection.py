from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from json import JSONDecodeError
import time
import json
import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
JSON_PATH = os.path.join(BASE_DIR, "output_Rushabh.json")       
CSV_PATH = os.path.join(BASE_DIR, "output_Rushabh.csv")

API_KEY = os.getenv("YOUTUBE_API_KEY")
VIDEO_ID = "NQypHE9_Fm4"

if not API_KEY:
    raise ValueError("API_KEY not found. Make sure YOUTUBE_API_KEY is set.")


youtube = build('youtube', 'v3', developerKey=API_KEY)

def fetch_all_comment_threads(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,        
        textFormat="plainText"
    )
    count = 0
    max_batches = 50 # edit this value to control how many comments to get

    while request and count < max_batches:
        commentAuthorIdSet = set()
        semicomments = []
        try:
            resp = request.execute()
        except HttpError as e:      # you may get an HttpError when making requests too fast or run out of quuota
           
            print("HttpError. Ending process")
            return comments
        
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            print("Found user: "+top.get("authorDisplayName"))
            comment = {
                "authorChannelId": top.get("authorChannelId", {}).get("value"), #is unpacked later
                "commentText": top.get("textDisplay"),
                "publishedAt": top.get("publishedAt"),
                "commentLikeCount": top.get("likeCount"),
            }
            semicomments.append(comment)
            commentAuthorIdSet.add(top.get("authorChannelId", {}).get("value"))
            
        commentAuthorIdSet = list(commentAuthorIdSet)
        channelIdDict = {}
        for i in range(0, len(commentAuthorIdSet), 50):
            batch = commentAuthorIdSet[i:i+50]
            try:
                channelresp = youtube.channels().list(
                    part="snippet,statistics",
                    id=",".join(batch),
                    maxResults=50
                ).execute()
            except HttpError:
                print("HttpError. Ending process")
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
        
        for item in semicomments:
            channelDict = channelIdDict[item["authorChannelId"]]
            comments.append({
                "channelTitle" : channelDict["title"],
                "channelDate" : channelDict["publishedAt"],
                "channelViewCount" : channelDict["viewCount"],
                "channelSubscriberCount" : channelDict["subscriberCount"],
                "channelVideoCount" : channelDict["videoCount"],
                "channelCountry" : channelDict["country"],
                "channelID" : item["authorChannelId"],
                "commentText" : item["commentText"],
                "commentDate" : item["publishedAt"],
                "commentLikeCount" : item["commentLikeCount"]
            })
        
        request = youtube.commentThreads().list_next(request, resp)
        count += 1
        time.sleep(0.1)
    
    return comments

if __name__ == "__main__":

    # Create output.json if missing
    if not os.path.exists(JSON_PATH):
        with open(JSON_PATH, "w") as f:
            json.dump([], f)

    # Load old comments safely
    try:
        with open(JSON_PATH, "r") as f:
            old_comments = json.load(f)
    except JSONDecodeError:
        print("output.json is corrupted - resetting file")
        old_comments = []
        with open(JSON_PATH, "w") as f:
            json.dump([], f)

    # Fetch new comments
    comments = fetch_all_comment_threads(VIDEO_ID)

    # Save combined results
    with open(JSON_PATH, "w") as f:
        json.dump(old_comments + comments, f, indent=2)
    
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved to {CSV_PATH}")
    print(df.head(1000))
    

    youtube.close()