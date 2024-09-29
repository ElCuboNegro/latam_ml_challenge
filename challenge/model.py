import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from datetime import datetime
import joblib
import os
from abc import ABC, abstractmethod
import logging
from typing import Union, Tuple
from sklearn.exceptions import NotFittedError
from imblearn.over_sampling import SMOTE

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
    def get_period_day(date: datetime) -> str:
        """
        Determina el período del día basado en la hora.

        Args:
            date (datetime): Fecha y hora.

        Returns:
            str: Período del día ('mañana', 'tarde', 'noche').
        """
        date_time = date.time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()


        if morning_min <= date_time <= morning_max:
            return "mañana"
        elif afternoon_min <= date_time <= afternoon_max:
            return "tarde"
        else:
            return "noche"

    @staticmethod
    def is_high_season(date: datetime) -> int:
        """
        Determina si una fecha está en temporada alta.

        Args:
            date (datetime): Fecha y hora.

        Returns:
            int: 1 si está en temporada alta, 0 en caso contrario.
        """
        fecha_año = date.year
        fecha_dt = date
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
        fecha_o = data["Fecha-O"]
        fecha_i = data["Fecha-I"]
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
        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=5.407
        )

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
        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=5.407
        )
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
            "OPERA_Copa Air"
        ]
        self.data_loader = CSVDataLoader()
        self.feature_processor = FeatureProcessor(self.top_10_features)
        self.date_processor = DateProcessor()
        self.model_persistence = ModelPersistence()

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:

        data["Fecha-I"] = pd.to_datetime(data["Fecha-I"])
        data["period_day"] = data["Fecha-I"].apply(
            self.date_processor.get_period_day
        )
        data["high_season"] = data["Fecha-I"].apply(
            self.date_processor.is_high_season
        )
        data["Fecha-O"] = pd.to_datetime(data["Fecha-O"])
        data["min_diff"] = (
            data["Fecha-O"] - data["Fecha-I"]
        ).dt.total_seconds() / 60
        data["delay"] = data["min_diff"].apply(
            lambda x: 1 if x > self.threshold_in_minutes else 0
        )

        # Generar variables dummy
        opera_dummies = pd.get_dummies(data["OPERA"], prefix="OPERA")
        tipo_vuelo_dummies = pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO")
        mes_dummies = pd.get_dummies(data["MES"], prefix="MES")

        # Asegurar que todas las columnas esperadas están presentes
        expected_columns = self.top_10_features

        for col in expected_columns:
            if col not in opera_dummies.columns and "OPERA_" in col:
                opera_dummies[col] = 0
            if col not in tipo_vuelo_dummies.columns and "TIPOVUELO_" in col:
                tipo_vuelo_dummies[col] = 0
            if col not in mes_dummies.columns and "MES_" in col:
                mes_dummies[col] = 0

        # Combinar las características y ordenarlas según expected_columns
        features = pd.concat(
            [
                mes_dummies[[col for col in expected_columns if col in mes_dummies.columns]],
                opera_dummies[[col for col in expected_columns if col in opera_dummies.columns]],
                tipo_vuelo_dummies[[col for col in expected_columns if col in tipo_vuelo_dummies.columns]],
            ],
            axis=1,
        )

        # Ordenar las columnas según expected_columns
        features = features[expected_columns]

        logging.info(f"Preprocessing completed. Features shape: {features.shape}")

        if target_column:
            target = data[[target_column]]  # Return target as DataFrame
            return features, target
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Entrena el modelo con las características y el objetivo proporcionados.

        Args:
            features (pd.DataFrame): Conjunto de características.
            target (pd.DataFrame): Variable objetivo.
        """
        logging.info("Iniciando el entrenamiento con SMOTE.")

        # Aplicar SMOTE para balancear las clases
        sm = SMOTE(random_state=42)
        features_resampled, target_resampled = sm.fit_resample(features, target)

        logging.info(f"Tamaño original de features: {features.shape}")
        logging.info(f"Tamaño después de SMOTE: {features_resampled.shape}")

        # Inicializar el modelo con los hiperparámetros ajustados
        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            use_label_encoder=False,
            eval_metric="logloss",
            max_depth=2,
            n_estimators=500,
            subsample=0.5,
            colsample_bytree=0.6,
            scale_pos_weight=1  # Ya no es necesario ajustar scale_pos_weight
        )

        # Entrenar el modelo con los datos balanceados
        self._model.fit(
            features_resampled, target_resampled.values.ravel()
        )

        # Guardar el modelo
        try:
            self.model_persistence.save_model(self._model)
            logging.info("Modelo guardado exitosamente.")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {e}")

    def get_confusion_matrix(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame
    ):
        """
        Obtiene la matriz de confusión del modelo.

        Args:
            X_test (pd.DataFrame): Conjunto de características de prueba.
            y_test (pd.DataFrame): Conjunto de etiquetas de prueba.

        Returns:
            np.ndarray: Matriz de confusión.
        """
        y_pred = self._model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return cm

    def predict(self, features: pd.DataFrame) -> list:
        """
        Realiza predicciones sobre las características proporcionadas.
        """
        # Asegurarse de que todas las características necesarias estén presentes
        for feature in self.top_10_features:
            if feature not in features.columns:
                features[feature] = 0

        # Seleccionar las características en el orden correcto
        features = features[self.top_10_features]
        features = features.astype(np.float64)

        try:
            # Realizar predicciones (0 o 1)
            predictions = self._model.predict(features)
        except NotFittedError:
            # Si el modelo no está entrenado, lo cargamos
            self._model = self.model_persistence.load_model()
            predictions = self._model.predict(features)
        except FileNotFoundError:
            # Si no hay modelo guardado, lanzamos una excepción clara
            raise Exception("El modelo no está entrenado y no se encontró un modelo guardado.")
        
        # Convertir el resultado a una lista y retornarlo
        return predictions.tolist()

    def get_report(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        Obtiene el reporte de clasificación del modelo.

        Args:
            X_test (pd.DataFrame): Conjunto de características de prueba.
            y_test (pd.DataFrame): Conjunto de etiquetas de prueba.

        Returns:
            dict: Reporte de clasificación.
        """
        y_pred = self._model.predict(X_test)
        self.report = classification_report(
            y_test, y_pred, output_dict=True
        )
        return self.report

def calculate_scale_pos_weight(target):
    n_y0 = len(target[target == 0])
    n_y1 = len(target[target == 1])
    return n_y0 / n_y1