import fastapi
import pandas as pd
from challenge import model
from typing import Dict, List, TypedDict

app = fastapi.FastAPI()
DelayModel = model.DelayModel()

class Flight(TypedDict):
    OPERA: str
    TIPOVUELO: str
    MES: int

class FlightData(TypedDict):
    flights: List[Flight]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(flight_data: FlightData) -> dict:
    flight = flight_data["flights"][0]
    operador = flight["OPERA"]
    tipo_vuelo = flight["TIPOVUELO"]
    mes = flight["MES"]

    if not (1 <= mes <= 12):
        raise fastapi.HTTPException(status_code=400, detail="Incorrect month, should be 1-12")
    
    ## Validación de tipo de vuelo
    if tipo_vuelo not in ['N', 'I']:
        raise fastapi.HTTPException(status_code=400, detail="Incorrect flight type, should be 'N' or 'I'")
    
    ## Validación de operador
    valid_operators = ['Aerolineas Argentinas', 'Grupo LATAM', 'Sky Airline', 'Copa Air']
    if operador not in valid_operators:
        raise fastapi.HTTPException(status_code=400, detail="Incorrect operator")
    
    flight_df = pd.DataFrame([flight])
    prediction = DelayModel.predict(flight_df)
    
    return {
        "predict": prediction
    }
