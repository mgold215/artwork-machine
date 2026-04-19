# metadata.py
# Generates YouTube video metadata (title, description, tags) from
# the CreativeDirection and AudioFeatures objects produced by the pipeline.
#
# All metadata is SEO-optimized for YouTube search discoverability.

# moodmixformat's social links — used in every video description
SPOTIFY_LINK = "https://open.spotify.com/artist/4rJNR9WTq9LLHBTH6jXkFf"
INSTAGRAM_LINK = "https://instagram.com/moodmixformat"
LINKTREE_LINK = "https://linktr.ee/moodmixformat"

# Base tags that appear on every upload
BASE_TAGS = [
    "official visualizer", "audio visualizer",
    "future house", "melodic house", "deep house",
    "electronic music", "house music", "new music 2026",
]

# YouTube max tag length is 500 characters total
MAX_TAG_CHARS = 500


def generate_metadata(
    artist: str,
    album: str,
    direction,
    features,
) -> dict:
    """Generate YouTube metadata from pipeline outputs.

    Args:
        artist: artist name (e.g., "moodmixformat")
        album: track/album name (e.g., "SIDEWAYZ")
        direction: CreativeDirection object from Claude
        features: AudioFeatures object from librosa analysis

    Returns:
        dict with keys: title, description, tags, category
    """
    title = f"{artist} - {album} (Official Visualizer)"

    # Build description with streaming links and metadata
    bpm = int(features.bpm) if features.bpm else ""
    key = features.key or ""
    tagline = direction.tagline if hasattr(direction, 'tagline') else ""
    mood_tags = getattr(features, 'mood_tags', [])
    genre_hints = getattr(features, 'genre_hints', [])

    description = f"""{album} by {artist} — official audio visualizer.

Stream on Spotify: {SPOTIFY_LINK}
Follow on Instagram: {INSTAGRAM_LINK}
Linktree: {LINKTREE_LINK}

Genre: {', '.join(genre_hints) if genre_hints else 'future house'}
BPM: {bpm}
Key: {key}

{tagline}

#{artist.replace(' ', '')} #{album.replace(' ', '')} #futurehouse #melodichouse #deephouse
#newmusic #electronicmusic #housemusic #officialvisualizer"""

    # Build tags list, respecting 500 char limit
    tags = [artist, album] + BASE_TAGS
    # Add genre hints and mood tags from audio analysis
    for tag in genre_hints + mood_tags:
        tags.append(tag)
    # Deduplicate while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        lower = tag.lower()
        if lower not in seen:
            seen.add(lower)
            unique_tags.append(tag)
    # Trim to fit within 500 char limit
    total = 0
    trimmed = []
    for tag in unique_tags:
        total += len(tag)
        if total > MAX_TAG_CHARS:
            break
        trimmed.append(tag)

    return {
        'title': title,
        'description': description,
        'tags': trimmed,
        'category': '10',  # Music
    }
