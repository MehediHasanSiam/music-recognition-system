import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SONGS_DIR = DATA_DIR / "songs"
FEATURE_DIR = DATA_DIR / "features"
DB_PATH = DATA_DIR / "music.db"
PLAYLIST_METADATA = DATA_DIR / "playlist_metadata.csv"

EVAL_DIR = ROOT / "evaluation"
RESULTS_DIR = ROOT / "results"

PLAYLIST_URL = "https://youtube.com/playlist?list=PLUpF17Buin-Gt_ELCHE4jgDhTCI6Pur6B&si=HIxSAxI-syG02m6N"

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

PEAK_NEIGHBORHOOD = (18, 18)
PEAK_DB_THRESHOLD = -42.0
MAX_PEAKS_PER_SECOND = 28
FANOUT = 12
MIN_HASH_DT = 1
MAX_HASH_DT = 160

CHROMA_FPS = 6.0
MIN_QUERY_SECONDS = 2.0

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

for p in [DATA_DIR, SONGS_DIR, FEATURE_DIR, EVAL_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
