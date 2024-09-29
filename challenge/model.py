import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import xgboost as xgb
from datetime import datetime
import joblib
import os
from abc import ABC, abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        pass


class CSVDataLoader(DataLoader):
    def load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo en {file_path}")
        return pd.read_csv(file_path, low_memory=False)


class DateProcessor:
    @staticmethod
    def get_period_day(date: str) -> str:
        """
        Determina el período del día basado en la hora.

        Args:
            date (str): Fecha y hora en formato 'YYYY-MM-DD HH:MM:SS'.

        Returns:
            str: Período del día ('mañana', 'tarde', 'noche').
        """
        date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", "%H:%M").time()
        evening_max = datetime.strptime("23:59", "%H:%M").time()
        night_min = datetime.strptime("00:00", "%H:%M").time()
        night_max = datetime.strptime("04:59", "%H:%M").time()

        if morning_min <= date_time <= morning_max:
            return "mañana"
        elif afternoon_min <= date_time <= afternoon_max:
            return "tarde"
        elif (
            evening_min <= date_time <= evening_max
            or night_min <= date_time <= night_max
        ):
            return "noche"

    @staticmethod
    def is_high_season(fecha: str) -> int:
        """
        Determina si una fecha está en temporada alta.

        Args:
            fecha (str): Fecha en formato 'YYYY-MM-DD HH:MM:SS'.

        Returns:
            int: 1 si está en temporada alta, 0 en caso contrario.
        """
        fecha_año = int(fecha.split("-")[0])
        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
        range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=fecha_año)
        range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=fecha_año)
        range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=fecha_año)
        range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=fecha_año)
        range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=fecha_año)
        range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=fecha_año)
        range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=fecha_año)
        range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=fecha_año)

        if (
            (range1_min <= fecha_dt <= range1_max)
            or (range2_min <= fecha_dt <= range2_max)
            or (range3_min <= fecha_dt <= range3_max)
            or (range4_min <= fecha_dt <= range4_max)
        ):
            return 1
        else:
            return 0

    @staticmethod
    def get_min_diff(data: pd.Series) -> float:
        """
        Calcula la diferencia en minutos entre 'Fecha-O' y 'Fecha-I'.

        Args:
            data (pd.Series): Serie con las columnas 'Fecha-O' y 'Fecha-I'.

        Returns:
            float: Diferencia en minutos.
        """
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = (fecha_o - fecha_i).total_seconds() / 60
        return min_diff


class FeatureProcessor:
    def __init__(self, top_features):
        self.top_features = top_features

    def process_features(self, data: pd.DataFrame, target_column: str = None):
        # Selección de las características
        if target_column:
            data = shuffle(
                data[["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "delay"]],
                random_state=111,
            )

        # Generación de variables dummy
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        features = features.astype(np.float64)

        if target_column:
            filtered_features = features[self.top_features]
            target_content = pd.DataFrame(data[target_column], columns=[target_column])
            return filtered_features, target_content

        if not target_column and len(features.columns) > len(self.top_features):
            features = features[self.top_features]

        return features


class ModelPersistence:
    """
    Clase para la persistencia del modelo.

    Este clase maneja el guardado y la carga del modelo de clasificación de retrasos de vuelos.

    Métodos:
        save_model(model): Guarda el modelo entrenado en el sistema de archivos.
        load_model(): Carga el modelo desde el sistema de archivos.
    """

    def __init__(self):
        self.model_path = "data/xgb_delay_model.pkl"

    def save_model(self, model):
        joblib.dump(model, self.model_path)
        print(f"Modelo guardado en {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en {self.model_path}"
        )


class DelayModel:
    """
    Clase DelayModel para la gestión y predicción de retrasos en vuelos.

    Atributos:
        _model: Modelo de clasificación utilizado para predecir retrasos.
        report: Reporte de métricas del modelo.
        threshold_in_minutes: Umbral de minutos para determinar si un vuelo está retrasado.
        top_10_features: Lista de las principales características utilizadas para el modelo.
        data_loader: Instancia de CSVDataLoader para cargar los datos.
        feature_processor: Instancia de FeatureProcessor para procesar las características.
        date_processor: Instancia de DateProcessor para procesar las fechas.

    Métodos:
        preprocess(data, target_column=None):
            Preprocesa los datos de entrada y, opcionalmente, la columna objetivo.

        fit(features, target):
            Entrena el modelo con las características y el objetivo proporcionados.

        predict(features):
            Realiza predicciones sobre las características proporcionadas.

        get_report():
            Obtiene el reporte de clasificación del modelo.

        tune_hyperparameters():
            Ajusta los hiperparámetros del modelo para optimizar su rendimiento.
    """

    def __init__(self):
        self._model = None
        self.report = None
        self.threshold_in_minutes = 15
        self.top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        self.data_loader = CSVDataLoader()
        self.feature_processor = FeatureProcessor(self.top_10_features)
        self.date_processor = DateProcessor()

    def preprocess(self, data: pd.DataFrame, target_column: str = None):
        """
        Preprocesa los datos de entrada y, opcionalmente, la columna objetivo.

        Args:
            data (pd.DataFrame): Datos de entrada.
            target_column (str, optional): Columna objetivo. Por defecto es None.

        Returns:
            pd.DataFrame: Datos preprocesados.
        """
        if "Fecha-O" in data.columns and "Fecha-I" in data.columns:
            # Aplicamos las transformaciones relacionadas con fechas
            data["period_day"] = data["Fecha-I"].apply(
                self.date_processor.get_period_day
            )
            data["high_season"] = data["Fecha-I"].apply(
                self.date_processor.is_high_season
            )
            data["min_diff"] = data.apply(self.date_processor.get_min_diff, axis=1)
            # Calculamos la columna 'delay'
            if target_column == "delay":
                data["delay"] = np.where(
                    data["min_diff"] > self.threshold_in_minutes, 0, 1
                )

        return self.feature_processor.process_features(data, target_column)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Entrena el modelo con XGBoost, incluyendo el balanceo de clases.
        """
        model_persistence = ModelPersistence()

        print("Entrenando el modelo XGBoost...")

        x_train, x_test, y_train, y_test = train_test_split(
            features[self.top_10_features], target, test_size=0.33, random_state=42
        )

        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0 / n_y1  # Cálculo para balancear las clases

        xgb_model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )
        xgb_model.fit(x_train, y_train)

        self.report = classification_report(
            y_test, xgb_model.predict(x_test), output_dict=True
        )
        model_persistence.save_model(xgb_model)
        self._model = xgb_model

    def get_confution_matrix(self):
        """
        Obtiene la matriz de confusión del modelo.

        Returns:
            np.ndarray: Matriz de confusión.
        """
        return self.report["confusion_matrix"]

    def predict(self, features: pd.DataFrame) -> list:
        """
        Realiza predicciones sobre las características proporcionadas.

        Args:
            features (pd.DataFrame): Características de entrada.

        Returns:
            list: Probabilidades de clasificación.
        """
        model_persistence = ModelPersistence()
        if self._model is None:
            self._model = model_persistence.load_model()

        for feature in self.top_10_features:
            if feature not in features.columns:
                features[feature] = 0
        features = features[self.top_10_features]

        # Features is a DataFrame with multiple rows, each row is a flight (10 cols x n rows)

        probabilities = []

        # We need to predict each row and return a list of lists

        # as the features are multiple rows, every one for a fligth,
        # we need to predict each row and return a list of lists
        # probabilities have 2 values that sum 1 (probabilities), first is for 0 and second for 1

        for index, row in features.iterrows():
            prob = self._model.predict_proba([row])[0].tolist()
            probabilities.append(prob)

        return probabilities

    def get_report(self):
        """
        Obtiene el reporte de clasificación del modelo.

        Returns:
            dict: Reporte de clasificación.
        """
        return self.report
