import json
from fastapi import FastAPI, UploadFile, File, HTTPException
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
    metadata_file: UploadFile = File(...)
):
    try:
        metadata_bytes = await metadata_file.read()
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {str(e)}")

    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="audio vacío")

        result = service.predict_bytes(audio_bytes)

        return {
                "estado": result["estado"],
                "precision": result["precision"],
                "scores": result["scores"],
                "limpieza": result["limpieza"]
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))