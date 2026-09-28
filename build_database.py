import argparse
import re

import numpy as np
import pandas as pd

from config import SONGS_DIR, FEATURE_DIR, PLAYLIST_METADATA, AUDIO_EXTENSIONS
from audio_utils import load_audio, duration_seconds
from features import extract_chroma_sequence, extract_global_feature
from fingerprint import fingerprint_audio
from database import initialize, rebuild, insert_song, insert_fingerprints

def load_metadata():
    if not PLAYLIST_METADATA.exists():
        return {}

    df = pd.read_csv(PLAYLIST_METADATA).fillna("")
    result = {}

    for _, row in df.iterrows():
        video_id = str(row.get("video_id", "")).strip()
        if video_id:
            result[video_id] = row.to_dict()

    return result

def parse_video_id(path):
    parts = path.stem.split("__", 2)
    return parts[1] if len(parts) >= 3 else ""

def fallback_title(path):
    stem = re.sub(r"^\d+__", "", path.stem)
    return stem.split("__", 1)[-1].replace("_", " ").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    con = rebuild() if args.rebuild else initialize()
    metadata = load_metadata()

    audio_files = sorted(
        p for p in SONGS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise SystemExit(f"No audio files found in {SONGS_DIR}")

    existing_paths = {
        row[0]
        for row in con.execute("SELECT path FROM songs").fetchall()
    }

    added = 0
    print(f"Found {len(audio_files)} audio files.")

    for idx, path in enumerate(audio_files, 1):
        if str(path) in existing_paths:
            print(f"[{idx}/{len(audio_files)}] Skip existing: {path.name}")
            continue

        print(f"[{idx}/{len(audio_files)}] Processing: {path.name}")

        try:
            y, sr = load_audio(path)
            duration = duration_seconds(y, sr)
            hashes = fingerprint_audio(y, sr)
            chroma = extract_chroma_sequence(y, sr)
            global_feature = extract_global_feature(y, sr)

            video_id = parse_video_id(path)
            md = metadata.get(video_id, {})

            title = str(md.get("title") or fallback_title(path))
            artist = str(md.get("channel") or md.get("uploader") or "Unknown")
            source_url = str(md.get("webpage_url") or "")
            thumbnail = str(md.get("thumbnail") or "")

            song_id = insert_song(
                con,
                video_id=video_id,
                title=title,
                artist=artist,
                path=path,
                duration=duration,
                source_url=source_url,
                thumbnail=thumbnail,
                chroma_path="",
                global_feature=global_feature,
            )

            chroma_path = FEATURE_DIR / f"song_{song_id:05d}.npz"
            np.savez_compressed(chroma_path, chroma=chroma)

            con.execute(
                "UPDATE songs SET chroma_path=? WHERE song_id=?",
                (str(chroma_path), song_id),
            )
            insert_fingerprints(con, song_id, hashes)
            con.commit()

            print(
                f"    song_id={song_id}, "
                f"hashes={len(hashes)}, "
                f"chroma_frames={chroma.shape[1]}"
            )
            added += 1

        except Exception as exc:
            con.rollback()
            print(f"    ERROR: {exc}")

    print(f"Done. Added {added} songs.")

if __name__ == "__main__":
    main()
