import argparse
import time

from clip_matcher import identify_clip
from hum_matcher import identify_hum
from recommender import recommend_similar

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clip", "hum"], required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    start = time.perf_counter()

    if args.mode == "clip":
        results = identify_clip(args.audio, args.top_k)
    else:
        results = identify_hum(args.audio, args.top_k)

    elapsed = time.perf_counter() - start

    print(f"Response time: {elapsed:.3f} s")

    if not results:
        print("No match found.")
        return

    for rank, item in enumerate(results, 1):
        print(
            f"{rank}. {item['title']} — {item['artist']} "
            f"| score={item['score']:.4f}"
        )

    print("\nRecommendations:")

    for item in recommend_similar(results[0]["song_id"], top_k=5):
        print(
            f"- {item['title']} — {item['artist']} "
            f"| similarity={item['similarity']:.4f}"
        )

if __name__ == "__main__":
    main()
