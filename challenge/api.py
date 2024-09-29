# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, root_validator, ValidationError
from typing import List
import pandas as pd
from datetime import datetime
from challenge.model import DelayModel
import numpy as np
import logging
import asyncio

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
model = DelayModel()

# Caché para almacenar el reporte
report_cache = {
    "data": None,
    "last_generated": None
}

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


class FlightsData(BaseModel):
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
    try:
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, model.get_report)
        return report
    except Exception as e:
        logging.error(f"Error al generar el reporte: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al generar el reporte.")

@app.post("/predict", status_code=200)
async def post_predict(data: FlightsData) -> dict:
    try:
        df = pd.DataFrame([flight.dict() for flight in data.flights])
        features = model.preprocess(df)
        predictions = model.predict(features)
        return {"predict": predictions}
    except Exception as e:
        return {"error": "An error occurred during prediction.", "detail": str(e)}
