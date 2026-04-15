# Heart Anomaly API

API para detectar si un sonido cardíaco es **normal** o **anormal** usando únicamente:

- `/home/Sonidos/Audios/normal/*.wav`
- `/home/Sonidos/Audios/anormal/*.wav`

No usa `ECG`, `ECG_1`, `ECG_2` ni JSON de entrenamiento.

## Qué hace

1. Lee audios `normal` y `anormal`.
2. Limpia el sonido con:
   - conversión a mono
   - recorte de silencios
   - filtro pasa banda cardíaco aproximado (25-400 Hz)
   - normalización
3. Extrae características acústicas.
4. Entrena un clasificador supervisado.
5. Expone una API en puerto `5004`.
6. En predicción recibe `audio` + `metadata_json`, analiza, responde JSON y borra el temporal.

## Estructura

- `app/audio_utils.py`: limpieza y features
- `app/model_service.py`: entrenamiento y predicción
- `app/main.py`: API FastAPI
- `docker-compose.yml`: despliegue

## Despliegue

Desde la carpeta del proyecto:

```bash
docker compose up -d --build
```

Ver logs:

```bash
docker compose logs -f
```

Probar salud:

```bash
curl http://127.0.0.1:5004/health
```

## Reentrenar

```bash
curl -X POST http://127.0.0.1:5004/retrain
```

## Predicción

```bash
curl -X POST "http://127.0.0.1:5004/predict" \
  -F 'audio=@/ruta/audio.wav' \
  -F 'metadata_json={"paciente":"123","origen":"prueba"}'
```

## Respuesta esperada

```json
{
  "status": "ok",
  "archivo": "audio.wav",
  "metadata_recibida": {
    "paciente": "123",
    "origen": "prueba"
  },
  "estado": "normal",
  "precision": 0.873421,
  "scores": {
    "anormal": 0.126579,
    "normal": 0.873421
  },
  "limpieza": {
    "sample_rate": 2000,
    "duration_seconds": 8.512
  }
}
```

## Notas importantes

- La `precision` devuelta es la **confianza del modelo** para esa predicción, no precisión clínica real.
- La precisión real del sistema depende mucho de cuántos audios etiquetados tengas y su calidad.
- Esto sirve como **clasificador inicial de normal vs anormal**. Para valvulopatías después habría que migrar a una clasificación multiclase o por hallazgos específicos.
