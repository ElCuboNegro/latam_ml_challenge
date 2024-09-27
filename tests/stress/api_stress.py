from locust import HttpUser, task
from pytest import mark


class StressUser(HttpUser):

    @task
    @mark.stress
    def predict_argentinas(self):
        self.client.post(
            "/predict",
            json={
                "flights": [
                    {"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}
                ]
            },
        )

    @task
    @mark.stress
    def predict_latam(self):
        self.client.post(
            "/predict",
            json={"flights": [{"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3}]},
        )
