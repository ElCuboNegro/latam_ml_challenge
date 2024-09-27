# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
import joblib
import os
from datetime import datetime

from challenge.model import DelayModel

app = FastAPI()


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightData(BaseModel):
    flights: List[Flight]


# Inicializar el modelo
delay_model = DelayModel()


# Cargar el modelo al iniciar la API
@app.on_event("startup")
def load_model_event():
    try:
        delay_model.load_model()
    except FileNotFoundError:
        print("Modelo no encontrado. Por favor, entrena el modelo primero.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/tune", status_code=200)
async def tune_hiperparameters() -> dict:
    delay_model.tune_hyperparameters()
    return {"status": "OK"}


@app.post("/predict", status_code=200)
def post_predict(flight_data: FlightData) -> dict:

    valid_operators = [
        "Aerolineas Argentinas",
        "Grupo LATAM",
        "Sky Airline",
        "Copa Air",
        "Latin American Wings",  # Asegúrate de incluir todos los operadores utilizados en el entrenamiento
    ]

    if not flight_data.flights:
        raise HTTPException(
            status_code=400, detail="No se proporcionaron vuelos para predecir."
        )

    flight_dict = flight_data.flights[0].dict()

    operador = flight_dict.get("OPERA")
    tipo_vuelo = flight_dict.get("TIPOVUELO")
    mes = flight_dict.get("MES")

    # mes 0 - 12
    if mes < 0 or mes > 12:
        raise HTTPException(
            status_code=400,
            detail=f"Mes incorrecto para vuelo {flight_dict}, debe estar entre 0 y 12.",
        )

    if tipo_vuelo not in ["N", "I"]:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de vuelo incorrecto para vuelo {flight_dict}, debe ser 'N' o 'I'.",
        )

    if operador not in valid_operators:
        raise HTTPException(
            status_code=400,
            detail=f"Operador incorrecto para vuelo {flight_dict}, debe ser uno de {valid_operators}.",
        )

    t_dict = {"OPERA": operador, "TIPOVUELO": tipo_vuelo, "MES": mes}

    flight_df = pd.DataFrame(t_dict, index=[0])

    # Preprocesar los datos
    try:
        features = delay_model.preprocess(flight_df, target_column=None)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en el preprocesamiento: {e}"
        )

    # Realizar predicciones
    try:
        predictions = delay_model.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")

    # Formatear las predicciones
    return predictions
