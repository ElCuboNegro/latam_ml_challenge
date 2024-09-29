# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, root_validator, ValidationError
from typing import List
import pandas as pd
from datetime import datetime
from challenge.model import DelayModel, ModelPersistence, CSVDataLoader
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("MES")
    def mes_must_be_valid(cls, v):
        if not 1 <= v <= 12:
            raise HTTPException(status_code=400, detail="MES debe estar entre 1 y 12")
        return v

    @validator("TIPOVUELO")
    def tipovuelo_must_be_valid(cls, v):
        if v not in ["N", "I"]:
            raise HTTPException(
                status_code=400,
                detail="TIPOVUELO debe ser 'N' (Nacional) o 'I' (Internacional)",
            )
        return v

    @validator("OPERA")
    def opera_must_be_valid(cls, v):
        valid_operators = [
            "American Airlines",
            "Air Canada",
            "Air France",
            "Aeromexico",
            "Aerolineas Argentinas",
            "Austral",
            "Avianca",
            "Alitalia",
            "British Airways",
            "Copa Air",
            "Delta Air",
            "Gol Trans",
            "Iberia",
            "K.L.M.",
            "Qantas Airways",
            "United Airlines",
            "Grupo LATAM",
            "Sky Airline",
            "Latin American Wings",
            "Plus Ultra Lineas Aereas",
            "JetSmart SPA",
            "Oceanair Linhas Aereas",
            "Lacsa",
        ]
        if v not in valid_operators:
            raise HTTPException(
                status_code=400, detail=f"OPERA debe ser uno de {valid_operators}"
            )
        return v


class FlightData(BaseModel):
    flights: List[Flight]

    @root_validator(pre=True)
    def check_flights_list_not_empty(cls, values):
        flights = values.get("flights")
        if not flights:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar al menos un vuelo para predecir.",
            )
        return values


# Inicializar el modelo
delay_model = DelayModel()
model_persistence = ModelPersistence()


# Cargar el modelo al iniciar la API
@app.on_event("startup")
def load_model_event():
    try:
        model_persistence.load_model()
    except FileNotFoundError:
        print("Modelo no encontrado. Por favor, entrena el modelo primero.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")


@app.get(
    "/health",
    status_code=200,
    description="Chequea el estado de la API",
    tags=["Health"],
)
async def get_health() -> dict:
    return {"status": "OK"}


@app.get(
    "/report",
    status_code=200,
    description="Obtiene el reporte de clasificación",
    tags=["Model"],
)
async def get_report() -> dict:
    return delay_model.get_report()


@app.post(
    "/tune",
    status_code=200,
    description="Tune model, puede tardar mucho tiempo",
    tags=["Model"],
)
async def fit_model() -> dict:
    data_loader = CSVDataLoader()
    data = data_loader.load_data("data/data.csv")
    features, target = delay_model.preprocess(data, target_column="delay")
    delay_model.fit(features, target)
    model_persistence.save_model(delay_model)
    return delay_model.get_report()


@app.post("/predict")
async def post_predict(flight_data: FlightData) -> dict:
    predictions = []
    for flight in flight_data.flights:
        try:
            logging.debug(f"Procesando vuelo: {flight}")
            flight_dict = flight.dict()
            flight_df = pd.DataFrame([flight_dict])
            features = delay_model.preprocess(flight_df, target_column=None)
            logging.debug(f"Características preprocesadas: {features}")
            prediction = delay_model.predict(features)
            logging.debug(f"Predicción cruda: {prediction}")
            prediction_value = float(prediction[0][0][0])
            predictions.append(1 if prediction_value > 0.5 else 0)
            logging.debug(f"Predicción final: {predictions[-1]}")
        except Exception as e:
            logging.error(f"Error en la predicción: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")

    return {"predict": predictions}
