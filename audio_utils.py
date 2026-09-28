from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

from config import SR

def load_audio(path, sr=SR):
    y, out_sr = librosa.load(path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")

    y, _ = librosa.effects.trim(y, top_db=35)
    if y.size == 0:
        raise ValueError(f"Audio became empty after trimming: {path}")

    peak = float(np.max(np.abs(y)))
    if peak > 1e-9:
        y = 0.95 * y / peak
    return y, out_sr

def duration_seconds(y, sr=SR):
    return float(len(y) / sr)

def save_wav(path, y, sr=SR):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(y, dtype=np.float32), sr)

def add_noise_at_snr(y, snr_db, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=np.float32)

    signal_power = np.mean(y ** 2) + 1e-12
    noise = rng.standard_normal(len(y)).astype(np.float32)
    noise_power = np.mean(noise ** 2) + 1e-12

    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(desired_noise_power / noise_power)

    out = y + noise
    peak = np.max(np.abs(out)) + 1e-12
    if peak > 1:
        out = out / peak
    return out.astype(np.float32)
