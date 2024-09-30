from locust import HttpUser, task, between
from locust.env import Environment

class StressUser(HttpUser):
    wait_time = between(1, 3)  # Tiempo de espera entre tareas

    @task
    def predict_flights(self):
        payloads = [
            {"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3},
            {"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3},
            # Puedes agregar más casos si es necesario
        ]
        for payload in payloads:
            with self.client.post("/predict", json={"flights": [payload]}, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Fallo en la petición con payload: {payload}")
                else:
                    response.success()
