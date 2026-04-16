from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .audio_utils import clean_heart_audio_from_bytes, extract_features
from .remote_dataset import get_remote_training_urls, download_audio


MODEL_PATH = Path("/app/model/model.joblib")


class HeartModelService:
    def __init__(self):
        self.model = None
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)

    def train_from_remote(self):
        normal_urls, anormal_urls = get_remote_training_urls()

        if not normal_urls or not anormal_urls:
            raise ValueError("No hay suficientes audios remotos en normal y anormal")

        X = []
        y = []
        errores = []

        for url in normal_urls:
            try:
                audio_bytes = download_audio(url)
                signal, sr = clean_heart_audio_from_bytes(audio_bytes)
                feats = extract_features(signal, sr)
                X.append(feats)
                y.append("normal")
            except Exception as e:
                errores.append({"url": url, "error": str(e)})

        for url in anormal_urls:
            try:
                audio_bytes = download_audio(url)
                signal, sr = clean_heart_audio_from_bytes(audio_bytes)
                feats = extract_features(signal, sr)
                X.append(feats)
                y.append("anormal")
            except Exception as e:
                errores.append({"url": url, "error": str(e)})

        if len(X) < 4:
            raise ValueError("Muy pocos audios válidos para entrenar")

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        stratify = y if len(set(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, preds, labels=["anormal", "normal"]).tolist()

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        self.model = pipeline

        return {
            "status": "ok",
            "dataset": {
                "total": int(len(X)),
                "normal": int((y == "normal").sum()),
                "anormal": int((y == "anormal").sum())
            },
            "metricas": {
                "accuracy": acc,
                "classification_report": report,
                "confusion_matrix": matrix
            },
            "errores_descarga_o_proceso": errores
        }

    def predict_bytes(self, audio_bytes: bytes):
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        signal, sr = clean_heart_audio_from_bytes(audio_bytes)
        feats = extract_features(signal, sr).reshape(1, -1)

        proba = self.model.predict_proba(feats)[0]
        classes = list(self.model.classes_)

        scores = {cls: float(p) for cls, p in zip(classes, proba)}

        prob_anormal = scores.get("anormal", 0.0)

        UMBRAL_NORMAL = 0.4
        UMBRAL_ANORMAL = 0.6

        if prob_anormal >= UMBRAL_ANORMAL:
            estado = "anormal"
            confidence = prob_anormal

        elif prob_anormal >= UMBRAL_NORMAL:
            estado = "sospechoso"
            confidence = prob_anormal

        else:
            estado = "normal"
            confidence = scores.get("normal", 0.0)

        return {
            "estado": estado,
            "precision": float(confidence),
            "umbral": UMBRAL,
            "scores": scores,
            "limpieza": {
                "sample_rate": sr,
                "duration_seconds": round(len(signal) / sr, 4)
            }
        }