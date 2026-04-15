from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt


TARGET_SR = 2000
MIN_DURATION_SECONDS = 2.0
MAX_DURATION_SECONDS = 20.0


def load_audio(path: str | Path, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32 and resample."""
    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("El audio está vacío")
    return y.astype(np.float32), loaded_sr


def bandpass_filter(signal: np.ndarray, sr: int, low: float = 25.0, high: float = 400.0, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * sr
    low_norm = max(low / nyquist, 1e-6)
    high_norm = min(high / nyquist, 0.999999)
    if low_norm >= high_norm:
        return signal
    b, a = butter(order, [low_norm, high_norm], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered.astype(np.float32)


def normalize_audio(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-8:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def trim_silence(signal: np.ndarray) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(signal, top_db=25)
    return trimmed.astype(np.float32) if trimmed.size else signal.astype(np.float32)


def clean_heart_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    y, sr = load_audio(path)
    y = trim_silence(y)
    y = bandpass_filter(y, sr)
    y = normalize_audio(y)
    duration = len(y) / sr
    if duration < MIN_DURATION_SECONDS:
        raise ValueError(f"Audio demasiado corto ({duration:.2f}s). Mínimo: {MIN_DURATION_SECONDS}s")
    if duration > MAX_DURATION_SECONDS:
        y = y[: int(MAX_DURATION_SECONDS * sr)]
    return y, sr


def save_clean_audio(signal: np.ndarray, sr: int, out_path: str | Path) -> None:
    sf.write(str(out_path), signal, sr)


def extract_features(signal: np.ndarray, sr: int) -> Dict[str, float]:
    features: Dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=signal)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)

    def add_stats(prefix: str, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float32).flatten()
        features[f"{prefix}_mean"] = float(np.mean(arr))
        features[f"{prefix}_std"] = float(np.std(arr))
        features[f"{prefix}_median"] = float(np.median(arr))
        features[f"{prefix}_p25"] = float(np.percentile(arr, 25))
        features[f"{prefix}_p75"] = float(np.percentile(arr, 75))

    for i in range(mfcc.shape[0]):
        add_stats(f"mfcc_{i+1}", mfcc[i])
        add_stats(f"mfcc_delta_{i+1}", delta[i])

    add_stats("zcr", zcr)
    add_stats("rms", rms)
    add_stats("centroid", centroid)
    add_stats("bandwidth", bandwidth)
    add_stats("rolloff", rolloff)
    add_stats("flatness", flatness)

    for i in range(contrast.shape[0]):
        add_stats(f"contrast_{i+1}", contrast[i])

    features["tempo"] = float(tempo)
    features["duration_seconds"] = float(len(signal) / sr)

    return features
