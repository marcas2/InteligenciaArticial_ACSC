import io
import numpy as np
import librosa
from scipy.signal import butter, filtfilt


TARGET_SR = 2000


def bandpass_filter(signal: np.ndarray, sr: int, lowcut: float = 20.0, highcut: float = 400.0) -> np.ndarray:
    nyquist = 0.5 * sr
    low = max(lowcut / nyquist, 1e-5)
    high = min(highcut / nyquist, 0.999)
    if low >= high:
        return signal
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)


def clean_heart_audio_from_bytes(audio_bytes: bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)

    if len(y) == 0:
        raise ValueError("Audio vacío")

    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) == 0:
        raise ValueError("Audio sin señal útil tras trim")

    y = bandpass_filter(y, sr, 20.0, 400.0)

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y.astype(np.float32), sr


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    feats = []

    for arr in [mfcc, spec_cent, spec_bw, rolloff, zcr, rms]:
        feats.extend(np.mean(arr, axis=1).tolist())
        feats.extend(np.std(arr, axis=1).tolist())

    return np.array(feats, dtype=np.float32)