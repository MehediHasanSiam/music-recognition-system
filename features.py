import numpy as np
import librosa

from config import SR, HOP_LENGTH, CHROMA_FPS

def _normalize_columns(X, eps=1e-8):
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    return X / np.maximum(norms, eps)

def _downsample_frames(X, source_fps, target_fps=CHROMA_FPS):
    if source_fps <= target_fps:
        return _normalize_columns(X)

    block = max(1, int(round(source_fps / target_fps)))
    n = X.shape[1] // block
    if n <= 0:
        return _normalize_columns(X)

    X = X[:, :n * block].reshape(X.shape[0], n, block).mean(axis=2)
    return _normalize_columns(X)

def extract_chroma_sequence(y, sr=SR):
    harmonic = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cens(
        y=harmonic,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    return _downsample_frames(chroma, sr / HOP_LENGTH)

def extract_global_feature(y, sr=SR):
    harmonic, percussive = librosa.effects.hpss(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_cens(y=harmonic, sr=sr, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=percussive, sr=sr)
    tempo = float(np.asarray(tempo).reshape(-1)[0])

    parts = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        chroma.mean(axis=1),
        chroma.std(axis=1),
        contrast.mean(axis=1),
        np.array([
            centroid.mean(), centroid.std(),
            bandwidth.mean(), bandwidth.std(),
            rolloff.mean(), rolloff.std(),
            zcr.mean(), zcr.std(),
            rms.mean(), rms.std(),
            tempo,
        ], dtype=np.float32),
    ]
    return np.nan_to_num(np.concatenate(parts).astype(np.float32))
