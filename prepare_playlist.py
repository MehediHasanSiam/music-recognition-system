import argparse

import pandas as pd
from yt_dlp import YoutubeDL

from config import PLAYLIST_URL, SONGS_DIR, PLAYLIST_METADATA

def extract_metadata(url):
    options = {
        "quiet": False,
        "extract_flat": True,
        "skip_download": True,
        "ignoreerrors": True,
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    rows = []

    for item in (info or {}).get("entries", []) or []:
        if not item:
            continue

        video_id = item.get("id", "")

        rows.append({
            "playlist_index": item.get("playlist_index", ""),
            "video_id": video_id,
            "title": item.get("title", ""),
            "channel": item.get("channel") or item.get("uploader") or "",
            "uploader": item.get("uploader", ""),
            "duration": item.get("duration", ""),
            "webpage_url": (
                item.get("url")
                if str(item.get("url", "")).startswith("http")
                else f"https://www.youtube.com/watch?v={video_id}"
                if video_id
                else ""
            ),
            "thumbnail": item.get("thumbnail", ""),
        })

    df = pd.DataFrame(rows)
    PLAYLIST_METADATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PLAYLIST_METADATA, index=False)
    print(f"Saved {len(df)} entries to {PLAYLIST_METADATA}")

def download_audio(url):
    SONGS_DIR.mkdir(parents=True, exist_ok=True)

    output = str(
        SONGS_DIR
        / "%(playlist_index)03d__%(id)s__%(title).180B.%(ext)s"
    )

    options = {
        "format": "bestaudio/best",
        "outtmpl": output,
        "ignoreerrors": True,
        "noplaylist": False,
        "restrictfilenames": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    with YoutubeDL(options) as ydl:
        ydl.download([url])

def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--metadata-only", action="store_true")
    group.add_argument("--download", action="store_true")
    ap.add_argument("--url", default=PLAYLIST_URL)
    args = ap.parse_args()

    extract_metadata(args.url)

    if args.download:
        print(
            "Downloading/converting audio. "
            "Use only material you are authorized to download/use."
        )
        download_audio(args.url)
    else:
        print(
            "Metadata only. Put authorized audio in data/songs/ "
            "or rerun with --download."
        )

if __name__ == "__main__":
    main()
