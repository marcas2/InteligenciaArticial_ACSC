from __future__ import annotations

import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .model_service import HeartSoundModelService


MODEL = HeartSoundModelService(
    base_data_dir=os.getenv("DATASET_BASE_DIR", "/data/Sonidos"),
    model_dir=os.getenv("MODEL_DIR", "/app/model_data"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL.load_or_train()
    yield


app = FastAPI(
    title="Heart Sound Anomaly API",
    version="1.0.0",
    description="API para detección básica de sonidos cardíacos anómalos usando solo audios normal/anormal.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL.pipeline is not None,
        "metadata": MODEL.metadata,
    }


@app.post("/retrain")
def retrain() -> dict:
    try:
        metadata = MODEL.train()
        return {"status": "ok", "message": "Modelo reentrenado", "metadata": metadata}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    metadata_json: str = Form(...),
):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    temp_path = None

    try:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"metadata_json no es un JSON válido: {exc}") from exc

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            if not content:
                raise HTTPException(status_code=400, detail="El archivo de audio está vacío")
            tmp.write(content)
            temp_path = tmp.name

        result = MODEL.predict_file(temp_path)
        return JSONResponse(
            content={
                "status": "ok",
                "archivo": audio.filename,
                "metadata_recibida": metadata,
                **result,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.get("/")
def root() -> dict:
    return {
        "service": "heart-anomaly-api",
        "version": "1.0.0",
        "endpoints": ["GET /health", "POST /predict", "POST /retrain"],
    }
