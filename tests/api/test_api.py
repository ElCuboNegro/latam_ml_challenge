import unittest
from fastapi.testclient import TestClient
from challenge import app
import numpy as np  # Agregar import para np si se necesita en el futuro
from typing import List, TypedDict



class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        # Simulación de respuesta del modelo
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    
    def test_should_fail_unknown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13  # Suponiendo que '13' no es válido
                }
            ]
        }
        # Simulación de error del modelo
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_fail_unknown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O",  # Suponiendo que 'O' no es válido
                    "MES": 13
                }
            ]
        }
        # Simulación de error del modelo
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
    
    def test_should_fail_unknown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas",  # Suponiendo que esta aerolínea no es válida
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # Simulación de error del modelo
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
