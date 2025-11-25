from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from json import JSONDecodeError
import time
import json


API_KEY = ""
VIDEO_ID = "fu6qi6R0qNs"


youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_genre(video_id):
    # 1. Get the video details
    video_resp = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    # extract categoryId
    category_id = video_resp["items"][0]["snippet"]["categoryId"]

    # 2. Convert categoryId -> readable genre
    cat_resp = youtube.videoCategories().list(
        part="snippet",
        id=category_id
    ).execute()

    genre = cat_resp["items"][0]["snippet"]["title"]
    return genre

def fetch_all_comment_threads(video_id, video_genre):
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
                thumb_url = item["snippet"]["thumbnails"]["default"]["url"]
                description = item["snippet"].get("description", "")

                channelIdDict[item["id"]] = {
                "title": item["snippet"]["title"],
                "publishedAt": item["snippet"]["publishedAt"],
                "viewCount": item["statistics"].get("viewCount"),
                "subscriberCount": item["statistics"].get("subscriberCount"),
                "videoCount": item["statistics"].get("videoCount"),
                "country": item["snippet"].get("country"),

                # NEW FEATURES
                "hasDescription": len(description.strip()) > 0,
                "defaultProfilePic": ("default" in thumb_url or "channels/default" in thumb_url)

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
                "commentLikeCount" : item["commentLikeCount"],
                "videoGenre": video_genre,
                "hasDescription": channelDict["hasDescription"],
                "defaultProfilePic": channelDict["defaultProfilePic"]
            })
        
        request = youtube.commentThreads().list_next(request, resp)
        count += 1
        time.sleep(0.1)
    
    return comments

if __name__ == "__main__":  # generates json file with comments for ANY video ID

    # Use single shared file
    OUTPUT_FILE = "output.json"

    # Load existing data if file exists
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            old_comments = json.load(f)
    except (JSONDecodeError, FileNotFoundError):
        old_comments = []

    # Fetch metadata for this video
    video_genre = get_video_genre(VIDEO_ID)

    # Scrape all comments + channel info
    comments = fetch_all_comment_threads(VIDEO_ID, video_genre)

    # Append NEW comments to the single global output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(old_comments + comments, f, indent=4, ensure_ascii=False)

    youtube.close()
    

