import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from database import connect, count_songs
from clip_matcher import identify_clip
from hum_matcher import identify_hum
from recommender import recommend_similar

st.set_page_config(
    page_title="Real-Time Music Recognition",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Real-Time Audio Pattern Recognition System")
st.caption(
    "Identify a song from a recorded clip or from humming/singing."
)

try:
    n_songs = count_songs(connect())
except Exception:
    n_songs = 0

st.sidebar.metric("Songs indexed", n_songs)
st.sidebar.markdown(
    "**Recognition modes**\n\n"
    "- **Audio clip:** landmark fingerprint + time-offset voting\n"
    "- **Humming / singing:** chroma + key-invariant subsequence DTW"
)

if n_songs == 0:
    st.warning(
        "Database is empty. Add songs and run "
        "`python build_database.py --rebuild`."
    )

mode_label = st.radio(
    "Query type",
    ["Audio clip", "Humming / singing"],
    horizontal=True,
)

source = st.radio(
    "Input source",
    ["Record microphone", "Upload file"],
    horizontal=True,
)

audio_data = None
suffix = ".wav"

if source == "Record microphone":
    audio_data = st.audio_input(
        "Record 5–15 seconds",
        sample_rate=22050,
    )
else:
    uploaded = st.file_uploader(
        "Upload a query",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
    )
    audio_data = uploaded

    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".wav"

if audio_data is not None:
    st.audio(audio_data)

if st.button(
    "🔎 Identify song",
    type="primary",
    disabled=(audio_data is None or n_songs == 0),
):
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:
            tmp.write(audio_data.getvalue())
            temp_path = tmp.name

        with st.spinner(
            "Analyzing audio and searching the database..."
        ):
            start = time.perf_counter()

            if mode_label == "Audio clip":
                results = identify_clip(temp_path, top_k=5)
            else:
                results = identify_hum(temp_path, top_k=5)

            elapsed = time.perf_counter() - start

        if not results:
            st.error("No candidate was found.")
        else:
            best = results[0]
            left, right = st.columns([2, 1])

            with left:
                st.subheader("Best match")
                st.write(f"### {best['title']}")
                st.write(
                    f"**Artist/channel:** {best['artist']}"
                )
                st.metric(
                    "Matching score",
                    f"{best['score']:.3f}",
                )
                st.metric(
                    "Search time",
                    f"{elapsed:.3f} s",
                )

                if mode_label == "Audio clip":
                    st.caption(
                        "Offset-consistent votes: "
                        f"{best.get('votes', 0)} | "
                        "Hash coverage: "
                        f"{best.get('coverage', 0):.4f}"
                    )
                else:
                    st.caption(
                        "DTW cost: "
                        f"{best.get('dtw_cost', 0):.4f} | "
                        "Best pitch-class shift: "
                        f"{best.get('pitch_shift', 0)} semitones"
                    )

            with right:
                if best.get("thumbnail"):
                    st.image(
                        best["thumbnail"],
                        caption="Source thumbnail",
                    )

                if best.get("source_url"):
                    st.link_button(
                        "Open source page",
                        best["source_url"],
                    )

            st.subheader("Ranked candidates")

            candidate_df = pd.DataFrame([
                {
                    "Rank": rank,
                    "Song": item["title"],
                    "Artist/channel": item["artist"],
                    "Score": round(item["score"], 4),
                }
                for rank, item in enumerate(results, 1)
            ])

            st.dataframe(
                candidate_df,
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("Similar songs")
            recs = recommend_similar(
                best["song_id"],
                top_k=5,
            )

            if recs:
                rec_df = pd.DataFrame([
                    {
                        "Song": item["title"],
                        "Artist/channel": item["artist"],
                        "Audio similarity": round(
                            item["similarity"], 4
                        ),
                    }
                    for item in recs
                ])

                st.dataframe(
                    rec_df,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info(
                    "Not enough indexed songs "
                    "for recommendations."
                )

    except Exception as exc:
        st.exception(exc)

st.divider()
st.caption(
    "Scores are algorithmic similarity measures, "
    "not calibrated probabilities. "
    "Use evaluation data to select an acceptance threshold."
)
