# youtube_auth.py
# One-time local setup script — NOT called by Railway.
# Opens your browser to authorize the YouTube app, then prints the refresh token.
#
# Prerequisites:
#   1. Create a Google Cloud project at https://console.cloud.google.com
#   2. Enable "YouTube Data API v3"
#   3. Create OAuth 2.0 credentials (Desktop app type)
#   4. Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in ~/.env.secrets
#
# Usage:
#   python3 -m artwork_machine.upload.youtube_auth
#
# After running:
#   1. Copy the YOUTUBE_REFRESH_TOKEN value into ~/.env.secrets
#   2. Add it to Railway environment variables
#   3. Run: source ~/.env.secrets

import os

from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes needed for uploading videos and setting thumbnails
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']


def main():
    client_id = os.environ.get('YOUTUBE_CLIENT_ID')
    client_secret = os.environ.get('YOUTUBE_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("ERROR: YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET must be set.")
        print("Set these in ~/.env.secrets. Run: source ~/.env.secrets")
        print()
        print("To get these values:")
        print("  1. Go to https://console.cloud.google.com")
        print("  2. Create a project (or select existing)")
        print("  3. Enable 'YouTube Data API v3'")
        print("  4. Go to Credentials -> Create Credentials -> OAuth 2.0 Client ID")
        print("  5. Application type: Desktop app")
        print("  6. Copy Client ID and Client Secret")
        return

    # Build the OAuth config from env vars (no client_secrets.json file needed)
    client_config = {
        'installed': {
            'client_id': client_id,
            'client_secret': client_secret,
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'redirect_uris': ['http://localhost'],
        }
    }

    print("Opening browser for YouTube authorization...")
    print("If browser doesn't open, visit the URL printed below.\n")

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    credentials = flow.run_local_server(port=8889)

    if not credentials or not credentials.refresh_token:
        print("ERROR: Authorization failed. Try again.")
        return

    print("\n" + "=" * 60)
    print("SUCCESS! Add this to ~/.env.secrets and Railway Variables:")
    print("=" * 60)
    print(f"\nYOUTUBE_REFRESH_TOKEN={credentials.refresh_token}")
    print(f"\nThen run: source ~/.env.secrets")
    print("Then add to Railway -> Service -> Variables")


if __name__ == '__main__':
    main()
