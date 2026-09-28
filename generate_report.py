import pandas as pd

from config import RESULTS_DIR
from database import connect, count_songs

def csv_table(path):
    if not path.exists():
        return None

    return pd.read_csv(path).to_markdown(
        index=False,
        floatfmt=".4f",
    )

def main():
    n_songs = count_songs(connect())

    clip_table = csv_table(
        RESULTS_DIR / "summary_clip.csv"
    )

    hum_table = csv_table(
        RESULTS_DIR / "summary_hum.csv"
    )

    lines = [
        "# Automatic Experimental Results Report",
        "",
        f"Indexed songs: **{n_songs}**",
        "",
        "## 1. System under test",
        "",
        "The system has two retrieval paths:",
        "",
        "1. Landmark audio fingerprinting with "
        "time-offset voting for recorded song clips.",
        "2. CENS chroma with 12-key transposition "
        "search and subsequence Dynamic Time Warping "
        "for humming/singing.",
        "",
        "Recommendations use standardized global "
        "audio descriptors and cosine similarity.",
        "",
        "## 2. Exact/recorded-clip evaluation",
        "",
    ]

    if clip_table:
        lines.extend([clip_table, ""])
    else:
        lines.extend([
            "No clip evaluation found.",
            "",
        ])

    lines.extend([
        "## 3. Human humming evaluation",
        "",
    ])

    if hum_table:
        lines.extend([hum_table, ""])
    else:
        lines.extend([
            "No human-humming evaluation found.",
            "",
        ])

    lines.extend([
        "## 4. Discussion checklist",
        "",
        "- Which condition produced the highest Top-1 accuracy?",
        "- How much did 10 dB noise change clip accuracy?",
        "- Why did exact fingerprinting degrade under pitch/time transformation?",
        "- How well did DTW tolerate humming key and tempo differences?",
        "- What was the mean response time?",
        "- Which songs were repeatedly confused?",
        "- What accuracy-latency-scalability trade-off was observed?",
        "",
        "## 5. Limitations",
        "",
        "- Polyphonic reference audio is not a clean symbolic melody.",
        "- Humming quality, microphone quality, noise, and query length matter.",
        "- Similarity scores are not calibrated probabilities.",
        "- Exhaustive DTW becomes slower as the database grows.",
    ])

    output = (
        RESULTS_DIR
        / "AUTO_RESULTS_REPORT.md"
    )

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Saved {output}")

if __name__ == "__main__":
    main()
