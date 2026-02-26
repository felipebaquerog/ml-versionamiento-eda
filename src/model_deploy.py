from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import os

# ===============================
# Cargar modelo
# ===============================

MODEL_PATH = "model/best_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Modelo no encontrado. Ejecuta train_model.py primero.")

model = joblib.load(MODEL_PATH)

# Guardar nombres de columnas esperadas
expected_features = model.feature_names_in_

# ===============================
# Crear API
# ===============================

app = FastAPI(
    title="Credit Risk Model API",
    description="API para predicción batch del modelo de riesgo crediticio",
    version="1.0"
)

class PredictionInput(BaseModel):
    data: List[dict]

@app.post("/predict")
def predict(input_data: PredictionInput):

    try:
        df = pd.DataFrame(input_data.data)

        # Validar columnas
        missing_cols = set(expected_features) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Faltan columnas: {missing_cols}"
            )

        # Ordenar columnas correctamente
        df = df[expected_features]

        predictions = model.predict(df)

        return {
            "n_registros": len(predictions),
            "predictions": predictions.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    





    