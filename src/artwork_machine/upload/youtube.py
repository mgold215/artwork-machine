# youtube.py
# Uploads a video to YouTube via the Data API v3 using resumable upload.
# Uses a refresh token from env (set up via youtube_auth.py one-time flow).
#
# Required environment variables:
#   YOUTUBE_CLIENT_ID      — Google Cloud OAuth client ID
#   YOUTUBE_CLIENT_SECRET  — Google Cloud OAuth client secret
#   YOUTUBE_REFRESH_TOKEN  — from running youtube_auth.py (one-time setup)

import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TOKEN_URI = 'https://oauth2.googleapis.com/token'
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']


def get_youtube_client():
    """Create and return an authenticated YouTube API client.

    Uses the refresh token from env to get fresh access tokens automatically.
    """
    credentials = Credentials(
        token=None,  # Will be refreshed automatically
        refresh_token=os.environ['YOUTUBE_REFRESH_TOKEN'],
        client_id=os.environ['YOUTUBE_CLIENT_ID'],
        client_secret=os.environ['YOUTUBE_CLIENT_SECRET'],
        token_uri=TOKEN_URI,
        scopes=SCOPES,
    )
    return build('youtube', 'v3', credentials=credentials)


def upload_video(
    video_path: str,
    metadata: dict,
    thumbnail_path: str = None,
) -> str:
    """Upload a video to YouTube with metadata.

    Args:
        video_path: path to the MP4 file
        metadata: dict with keys: title, description, tags, category
        thumbnail_path: optional path to thumbnail image (1280x720)

    Returns:
        YouTube video URL (e.g., "https://youtu.be/abc123") or None on failure
    """
    try:
        youtube = get_youtube_client()
    except KeyError as e:
        print(f"  YouTube upload skipped: {e} not set in environment")
        return None

    body = {
        'snippet': {
            'title': metadata['title'],
            'description': metadata['description'],
            'tags': metadata.get('tags', []),
            'categoryId': metadata.get('category', '10'),
        },
        'status': {
            'privacyStatus': 'public',
            'selfDeclaredMadeForKids': False,
        },
    }

    # Use resumable upload for reliability with large video files
    media = MediaFileUpload(
        video_path,
        mimetype='video/mp4',
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10 MB chunks
    )

    print(f"  Uploading to YouTube: {metadata['title']}...")

    request = youtube.videos().insert(
        part='snippet,status',
        body=body,
        media_body=media,
    )

    # Upload in chunks with progress reporting
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"  Upload progress: {pct}%")

    video_id = response['id']
    video_url = f"https://youtu.be/{video_id}"
    print(f"  Upload complete: {video_url}")

    # Set custom thumbnail if provided
    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype='image/png'),
            ).execute()
            print("  Custom thumbnail set")
        except Exception as e:
            print(f"  WARNING: Could not set thumbnail: {e}")

    return video_url
