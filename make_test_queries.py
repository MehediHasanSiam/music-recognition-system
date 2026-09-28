import argparse

import librosa
import numpy as np

from config import EVAL_DIR, SR
from database import connect, list_songs
from audio_utils import load_audio, save_wav, add_noise_at_snr

def choose_starts(n_samples, query_len, count):
    if n_samples <= query_len:
        return [0]

    low = min(
        int(5 * SR),
        max(0, n_samples - query_len),
    )

    high = max(
        low,
        n_samples - query_len - int(5 * SR),
    )

    if high <= low:
        return [low]

    return np.linspace(
        low,
        high,
        num=count,
        dtype=int,
    ).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queries-per-song",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--query-seconds",
        type=float,
        default=10.0,
    )
    args = ap.parse_args()

    out_root = EVAL_DIR / "generated"
    rng = np.random.default_rng(42)
    songs = list_songs(connect())

    if not songs:
        raise SystemExit("No songs in database.")

    q_len = int(args.query_seconds * SR)

    for song in songs:
        y, sr = load_audio(song["path"])

        starts = choose_starts(
            len(y),
            q_len,
            args.queries_per_song,
        )

        for query_index, start in enumerate(starts, 1):
            clip = y[start:start + q_len]

            if len(clip) < int(2 * SR):
                continue

            variants = {
                "clean": clip,
                "noise_10db": add_noise_at_snr(
                    clip,
                    10,
                    rng,
                ),
                "pitch_plus2": librosa.effects.pitch_shift(
                    clip,
                    sr=SR,
                    n_steps=2,
                ),
                "tempo_085": librosa.effects.time_stretch(
                    clip,
                    rate=0.85,
                ),
                "tempo_115": librosa.effects.time_stretch(
                    clip,
                    rate=1.15,
                ),
            }

            for condition, audio in variants.items():
                path = (
                    out_root
                    / condition
                    / str(song["song_id"])
                    / f"q{query_index:02d}.wav"
                )
                save_wav(path, audio, SR)

    print(f"Generated test queries in {out_root}")

if __name__ == "__main__":
    main()
