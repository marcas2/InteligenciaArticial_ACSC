import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from .model_service import HeartModelService

app = FastAPI(title="Heart Anomaly API", version="1.0.0")
service = HeartModelService()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo_cargado": service.model is not None
    }


@app.post("/retrain")
def retrain():
    try:
        result = service.train_from_remote()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    metadata_json: str = Form(...)
):
    try:
        metadata = json.loads(metadata_json)
    except Exception:
        raise HTTPException(status_code=400, detail="metadata_json inválido")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="audio vacío")

        result = service.predict_bytes(audio_bytes)

        return {
            "status": "ok",
            "archivo": audio.filename,
            "metadata_recibida": metadata,
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))