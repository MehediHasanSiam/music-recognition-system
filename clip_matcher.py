from audio_utils import load_audio
from fingerprint import fingerprint_audio, rank_hash_matches
from database import connect, fetch_matching_hash_rows, get_song

def identify_clip(audio_path, top_k=5):
    y, sr = load_audio(audio_path)
    query_hashes = fingerprint_audio(y, sr)

    if not query_hashes:
        return []

    con = connect()
    rows = fetch_matching_hash_rows(con, query_hashes)
    ranked = rank_hash_matches(query_hashes, rows)

    results = []

    for item in ranked[:top_k]:
        song = get_song(con, item["song_id"])
        if song is None:
            continue

        # Display score only; use raw coverage/votes for analysis.
        score = min(1.0, item["coverage"] * 8.0)

        results.append({
            "song_id": int(song["song_id"]),
            "title": song["title"],
            "artist": song["artist"],
            "score": float(score),
            "votes": int(item["votes"]),
            "coverage": float(item["coverage"]),
            "thumbnail": song["thumbnail"],
            "source_url": song["source_url"],
        })

    return results
