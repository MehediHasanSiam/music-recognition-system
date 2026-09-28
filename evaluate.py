import argparse
import time
from pathlib import Path

import pandas as pd

from config import EVAL_DIR, RESULTS_DIR
from clip_matcher import identify_clip
from hum_matcher import identify_hum

def iter_generated():
    root = EVAL_DIR / "generated"

    if not root.exists():
        return

    for condition_dir in sorted(
        p for p in root.iterdir() if p.is_dir()
    ):
        for song_dir in sorted(
            p for p in condition_dir.iterdir() if p.is_dir()
        ):
            try:
                truth = int(song_dir.name)
            except ValueError:
                continue

            for audio in sorted(
                song_dir.glob("*.wav")
            ):
                yield condition_dir.name, truth, audio

def iter_hum(hum_dir):
    root = Path(hum_dir)

    if not root.exists():
        return

    for song_dir in sorted(
        p for p in root.iterdir() if p.is_dir()
    ):
        try:
            truth = int(song_dir.name)
        except ValueError:
            continue

        for audio in sorted(song_dir.glob("*")):
            if audio.suffix.lower() in {
                ".wav", ".mp3", ".flac", ".m4a", ".ogg"
            }:
                yield "human_hum", truth, audio

def reciprocal_rank(results, truth):
    for rank, item in enumerate(results, 1):
        if int(item["song_id"]) == int(truth):
            return 1.0 / rank
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["clip", "hum"],
        required=True,
    )
    ap.add_argument(
        "--generated",
        action="store_true",
    )
    ap.add_argument(
        "--hum-dir",
        default=str(EVAL_DIR / "hum"),
    )
    args = ap.parse_args()

    if args.generated:
        items = list(iter_generated() or [])
    else:
        items = list(iter_hum(args.hum_dir) or [])

    if not items:
        raise SystemExit("No evaluation queries found.")

    records = []

    for condition, truth, audio in items:
        start = time.perf_counter()

        try:
            if args.mode == "clip":
                results = identify_clip(audio, 5)
            else:
                results = identify_hum(audio, 5)

            elapsed = time.perf_counter() - start
            ranked_ids = [
                int(item["song_id"])
                for item in results
            ]

            records.append({
                "condition": condition,
                "truth_song_id": truth,
                "query": str(audio),
                "predicted_song_id": (
                    ranked_ids[0]
                    if ranked_ids
                    else None
                ),
                "top1_correct": int(
                    bool(ranked_ids)
                    and ranked_ids[0] == truth
                ),
                "top3_correct": int(
                    truth in ranked_ids[:3]
                ),
                "mrr": reciprocal_rank(
                    results,
                    truth,
                ),
                "latency_s": elapsed,
                "best_score": (
                    results[0]["score"]
                    if results
                    else 0.0
                ),
                "error": "",
            })

        except Exception as exc:
            records.append({
                "condition": condition,
                "truth_song_id": truth,
                "query": str(audio),
                "predicted_song_id": None,
                "top1_correct": 0,
                "top3_correct": 0,
                "mrr": 0.0,
                "latency_s": None,
                "best_score": 0.0,
                "error": str(exc),
            })

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_path = (
        RESULTS_DIR
        / f"evaluation_{args.mode}.csv"
    )

    df.to_csv(
        detail_path,
        index=False,
    )

    summary = (
        df.groupby(
            "condition",
            dropna=False,
        )
        .agg(
            queries=("query", "count"),
            top1_accuracy=(
                "top1_correct",
                "mean",
            ),
            top3_accuracy=(
                "top3_correct",
                "mean",
            ),
            mrr=("mrr", "mean"),
            mean_latency_s=(
                "latency_s",
                "mean",
            ),
            mean_best_score=(
                "best_score",
                "mean",
            ),
        )
        .reset_index()
    )

    overall = pd.DataFrame([{
        "condition": "OVERALL",
        "queries": len(df),
        "top1_accuracy": df["top1_correct"].mean(),
        "top3_accuracy": df["top3_correct"].mean(),
        "mrr": df["mrr"].mean(),
        "mean_latency_s": df["latency_s"].mean(),
        "mean_best_score": df["best_score"].mean(),
    }])

    summary = pd.concat(
        [summary, overall],
        ignore_index=True,
    )

    summary_path = (
        RESULTS_DIR
        / f"summary_{args.mode}.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(summary.to_string(index=False))
    print(
        f"\nSaved:\n"
        f"- {detail_path}\n"
        f"- {summary_path}"
    )

if __name__ == "__main__":
    main()
