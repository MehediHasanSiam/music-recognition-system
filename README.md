# Real-Time Audio Pattern Recognition System for Music

This package implements the complete baseline system required by the supplied Complex Engineering Problem:

- offline song-database preparation
- audio preprocessing
- landmark audio fingerprinting for exact/recorded song clips
- chroma + transposition-invariant subsequence DTW for humming/singing queries
- SQLite fingerprint database
- content-based similar-song recommendation
- Streamlit GUI with microphone recording and file upload
- controlled experiment generation
- automatic evaluation and result export
- report/presentation/demo guidance

Default playlist:
`https://youtube.com/playlist?list=PLUpF17Buin-Gt_ELCHE4jgDhTCI6Pur6B&si=HIxSAxI-syG02m6N`

> Use playlist downloading only for audio that you are authorized to download/use. You can also place legally obtained audio files directly in `data/songs/`.

## 1. Install Python and FFmpeg

Recommended: Python 3.11 or 3.12.

FFmpeg is required if playlist audio must be converted to WAV.

```bash
python --version
ffmpeg -version
```

## 2. Create the environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Read the playlist metadata

```bash
python prepare_playlist.py --metadata-only
```

This creates `data/playlist_metadata.csv`.

## 4. Add the song audio

### Playlist route

Only for audio you are authorized to download/use:

```bash
python prepare_playlist.py --download
```

### Local-file route

Put `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, or `.aac` files in `data/songs/`.

## 5. Build the searchable database

```bash
python build_database.py --rebuild
```

This creates:

- `data/music.db`
- `data/features/song_XXXXX.npz`

The builder performs:

1. mono conversion/resampling
2. silence trimming and normalization
3. landmark fingerprint generation for recorded clips
4. CENS-chroma extraction for humming/singing
5. global audio-feature extraction for recommendations
6. SQLite indexing

## 6. Test from the command line

Recorded song clip:

```bash
python identify.py --mode clip --audio path/to/query.wav
```

Humming/singing:

```bash
python identify.py --mode hum --audio path/to/hum.wav
```

## 7. Run the GUI

```bash
streamlit run app.py
```

The GUI supports:

- microphone recording
- file upload
- Audio clip mode
- Humming/singing mode
- Top-5 candidate ranking
- recommendation
- response time
- source thumbnail/link when metadata is available

## 8. Generate controlled test queries

```bash
python make_test_queries.py --queries-per-song 3 --query-seconds 10
```

Conditions:

- `clean`
- `noise_10db`
- `pitch_plus2`
- `tempo_085`
- `tempo_115`

## 9. Evaluate recorded-clip recognition

```bash
python evaluate.py --mode clip --generated
```

Outputs:

- `results/evaluation_clip.csv`
- `results/summary_clip.csv`

Metrics:

- Top-1 accuracy
- Top-3 accuracy
- MRR
- mean latency
- mean best score
- results by condition

## 10. Evaluate real humming

First list song IDs:

```bash
python list_songs.py
```

Then record human humming files into:

```text
evaluation/
  hum/
    1/
      hum1.wav
      hum2.wav
    2/
      hum1.wav
      hum2.wav
```

Run:

```bash
python evaluate.py --mode hum --hum-dir evaluation/hum
```

Outputs:

- `results/evaluation_hum.csv`
- `results/summary_hum.csv`

For a defensible project experiment, collect 2–3 queries for at least 10 songs and, if possible, use more than one person.

## 11. Generate the measured-results report

```bash
python generate_report.py
```

Output:

- `results/AUTO_RESULTS_REPORT.md`

The report generator does not fabricate metrics.

## 12. Demo sequence

1. Show the playlist/source.
2. Show indexed-song count.
3. Identify a real song clip.
4. Show Top-5 ranking and latency.
5. Record 8–12 seconds of humming.
6. Identify the humming.
7. Show recommendations.
8. Show measured evaluation tables.
9. Explain failure cases and limitations.

See `docs/DEMO_SCRIPT.md`.

## System architecture

```text
OFFLINE
Playlist / local songs
        |
        +--> audio preprocessing
        |
        +--> landmark hashes ----------> SQLite fingerprint index
        |
        +--> CENS chroma --------------> per-song feature files
        |
        +--> global audio descriptors -> recommendation vectors

ONLINE: RECORDED CLIP
query -> fingerprint -> hash lookup -> time-offset voting -> ranked songs

ONLINE: HUMMING
query -> chroma -> 12 pitch shifts -> subsequence DTW -> ranked songs

OUTPUT
best match -> metadata -> similar songs -> GUI
```

## Important design decision

This project deliberately uses two recognition methods:

- **Recorded original clip:** sparse landmark fingerprinting.
- **Humming/singing:** chroma + transposition search + subsequence DTW.

Humming should not be forced through the exact-fingerprint engine because a human voice is not a copy of the studio recording.

## What must come from real execution

These cannot be honestly pre-filled in advance:

- actual playlist audio
- human humming recordings
- measured accuracy
- measured response time on your computer
- GUI screenshots
- final numerical discussion/conclusion

The package contains the scripts needed to produce those outputs.
