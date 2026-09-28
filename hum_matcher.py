from pathlib import Path

import numpy as np
import librosa

from audio_utils import load_audio
from features import extract_chroma_sequence
from database import connect, list_songs
from config import MIN_QUERY_SECONDS

def _safe_subsequence_cost(query, reference):
    if query.shape[1] < 2 or reference.shape[1] < 2:
        return float("inf")

    if reference.shape[1] < query.shape[1]:
        D = librosa.sequence.dtw(
            X=query,
            Y=reference,
            metric="cosine",
            backtrack=False,
        )
        return float(D[-1, -1] / max(1, query.shape[1]))

    D = librosa.sequence.dtw(
        X=query,
        Y=reference,
        metric="cosine",
        subseq=True,
        backtrack=False,
    )

    return float(np.nanmin(D[-1, :]) / max(1, query.shape[1]))

def _best_transposition_cost(query, reference):
    best_cost = float("inf")
    best_shift = 0

    for shift in range(12):
        shifted_query = np.roll(query, shift=shift, axis=0)
        cost = _safe_subsequence_cost(shifted_query, reference)

        if cost < best_cost:
            best_cost = cost
            best_shift = shift

    return best_cost, best_shift

def identify_hum(audio_path, top_k=5):
    y, sr = load_audio(audio_path)

    if len(y) / sr < MIN_QUERY_SECONDS:
        raise ValueError(
            f"Please provide at least {MIN_QUERY_SECONDS:.0f} seconds "
            "of humming/singing."
        )

    query = extract_chroma_sequence(y, sr)
    songs = list_songs(connect())
    ranked = []

    for song in songs:
        chroma_path = song["chroma_path"]

        if not chroma_path or not Path(chroma_path).exists():
            continue

        reference = np.load(chroma_path)["chroma"].astype(np.float32)
        cost, shift = _best_transposition_cost(query, reference)
        similarity = float(np.exp(-max(0.0, cost)))

        ranked.append({
            "song_id": int(song["song_id"]),
            "title": song["title"],
            "artist": song["artist"],
            "score": similarity,
            "dtw_cost": float(cost),
            "pitch_shift": int(shift),
            "thumbnail": song["thumbnail"],
            "source_url": song["source_url"],
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]
