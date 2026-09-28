import hashlib
from collections import defaultdict

import numpy as np
import librosa
from scipy.ndimage import maximum_filter

from config import (
    SR, N_FFT, HOP_LENGTH, PEAK_NEIGHBORHOOD, PEAK_DB_THRESHOLD,
    MAX_PEAKS_PER_SECOND, FANOUT, MIN_HASH_DT, MAX_HASH_DT
)

def spectrogram_peaks(y, sr=SR):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    local_max = maximum_filter(S_db, size=PEAK_NEIGHBORHOOD, mode="nearest")
    mask = (S_db == local_max) & (S_db >= PEAK_DB_THRESHOLD)
    freq_idx, time_idx = np.where(mask)

    if len(freq_idx) == 0:
        return []

    strengths = S_db[freq_idx, time_idx]
    duration = max(len(y) / sr, 1.0)
    max_count = max(1, int(MAX_PEAKS_PER_SECOND * duration))

    if len(strengths) > max_count:
        keep = np.argpartition(strengths, -max_count)[-max_count:]
        freq_idx = freq_idx[keep]
        time_idx = time_idx[keep]
        strengths = strengths[keep]

    return sorted(
        zip(time_idx.astype(int), freq_idx.astype(int), strengths.astype(float)),
        key=lambda x: (x[0], -x[2])
    )

def _hash_triplet(f1, f2, dt):
    return hashlib.sha1(f"{f1}|{f2}|{dt}".encode("utf-8")).hexdigest()[:20]

def fingerprint_audio(y, sr=SR):
    peaks = spectrogram_peaks(y, sr)
    hashes = []

    for i, (t1, f1, _) in enumerate(peaks):
        paired = 0
        for j in range(i + 1, len(peaks)):
            t2, f2, _ = peaks[j]
            dt = t2 - t1

            if dt < MIN_HASH_DT:
                continue
            if dt > MAX_HASH_DT:
                break

            hashes.append((_hash_triplet(f1, f2, dt), int(t1)))
            paired += 1
            if paired >= FANOUT:
                break

    return hashes

def rank_hash_matches(query_hashes, rows):
    q_by_hash = defaultdict(list)
    for h, q_offset in query_hashes:
        q_by_hash[h].append(q_offset)

    votes = defaultdict(lambda: defaultdict(int))
    raw_matches = defaultdict(int)

    for h, song_id, db_offset in rows:
        for q_offset in q_by_hash.get(h, []):
            delta = int(db_offset) - int(q_offset)
            votes[int(song_id)][delta] += 1
            raw_matches[int(song_id)] += 1

    ranked = []
    q_count = max(1, len(query_hashes))

    for song_id, hist in votes.items():
        best_votes = max(hist.values()) if hist else 0
        ranked.append({
            "song_id": song_id,
            "votes": best_votes,
            "coverage": float(best_votes / q_count),
            "raw_matches": raw_matches[song_id],
        })

    ranked.sort(key=lambda x: (x["votes"], x["raw_matches"]), reverse=True)
    return ranked
