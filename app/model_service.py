from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audio_utils import clean_heart_audio, extract_features


class HeartSoundModelService:
    def __init__(self, base_data_dir: str | Path = "/data/Sonidos", model_dir: str | Path = "/app/model_data") -> None:
        self.base_data_dir = Path(base_data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "heart_anomaly_model.joblib"
        self.meta_path = self.model_dir / "heart_anomaly_model_meta.json"
        self.pipeline: Pipeline | None = None
        self.feature_columns: List[str] = []
        self.metadata: Dict[str, object] = {}

    def train(self) -> Dict[str, object]:
        rows: List[Dict[str, float]] = []
        labels: List[str] = []
        processed_files: List[str] = []
        skipped_files: List[Tuple[str, str]] = []

        folders = {
            "normal": self.base_data_dir / "Audios" / "normal",
            "anormal": self.base_data_dir / "Audios" / "anormal",
        }

        for label, folder in folders.items():
            if not folder.exists():
                continue
            for wav_path in sorted(folder.glob("*.wav")):
                try:
                    signal, sr = clean_heart_audio(wav_path)
                    feats = extract_features(signal, sr)
                    rows.append(feats)
                    labels.append(label)
                    processed_files.append(str(wav_path))
                except Exception as exc:
                    skipped_files.append((str(wav_path), str(exc)))

        if len(rows) < 6:
            raise RuntimeError(
                "No hay suficientes audios válidos para entrenar. Se requieren al menos 6 entre normal y anormal."
            )

        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            raise RuntimeError("Se requieren ambas clases: normal y anormal.")

        X = pd.DataFrame(rows)
        y = np.array(labels)
        self.feature_columns = list(X.columns)

        stratify = y if min(np.unique(y, return_counts=True)[1]) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=16,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        joblib.dump({"pipeline": pipeline, "feature_columns": self.feature_columns}, self.model_path)
        self.metadata = {
            "accuracy_holdout": accuracy,
            "report": report,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "processed_files": len(processed_files),
            "skipped_files": skipped_files,
            "labels": unique_labels,
        }
        self.meta_path.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        self.pipeline = pipeline
        return self.metadata

    def load_or_train(self) -> Dict[str, object]:
        if self.model_path.exists():
            data = joblib.load(self.model_path)
            self.pipeline = data["pipeline"]
            self.feature_columns = list(data["feature_columns"])
            if self.meta_path.exists():
                self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return self.metadata
        return self.train()

    def predict_file(self, audio_path: str | Path) -> Dict[str, object]:
        if self.pipeline is None:
            self.load_or_train()
        if self.pipeline is None:
            raise RuntimeError("El modelo no está cargado")

        signal, sr = clean_heart_audio(audio_path)
        feats = extract_features(signal, sr)
        row = pd.DataFrame([feats])
        row = row.reindex(columns=self.feature_columns, fill_value=0.0)
        pred = self.pipeline.predict(row)[0]
        proba = self.pipeline.predict_proba(row)[0]
        classes = list(self.pipeline.named_steps["clf"].classes_)
        class_scores = {label: float(score) for label, score in zip(classes, proba)}
        confidence = float(np.max(proba))

        return {
            "estado": pred,
            "precision": round(confidence, 6),
            "scores": class_scores,
            "limpieza": {
                "sample_rate": sr,
                "duration_seconds": round(len(signal) / sr, 3),
            },
        }
