import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from datetime import datetime
import joblib
import os
import matplotlib.pyplot as plt


class DelayModel:
    def __init__(self):
        """
        Inicializa la clase DelayModel.

        Args:
            model_path (str): Ruta donde se guardará el modelo entrenado.
        """
        self._model = None
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
        self.model_path = "data/xgb_delay_model.pkl"
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.scale_pos_weight = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga los datos desde un archivo CSV.

        Args:
            file_path (str): Ruta al archivo CSV.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo en {file_path}")
        data = pd.read_csv(file_path)
        return data

    def get_period_day(self, date: str) -> str:
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

    def is_high_season(self, fecha: str) -> int:
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

    def get_min_diff(self, data: pd.Series) -> float:
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

    def preprocess(self, data: pd.DataFrame, target_column: str = None):
        """
        Preprocesa los datos para el modelo.

        Args:
            data (pd.DataFrame): Datos originales.
            target_column (str, optional): Nombre de la columna objetivo. Defaults to None.

        Returns:
            Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]: Características y target si se especifica.
        """

        if (
            "Fecha-O" in data.columns
            and "Fecha-I" in data.columns
            and target_column == "delay"
        ):
            data["period_day"] = data["Fecha-I"].apply(self.get_period_day)
            data["high_season"] = data["Fecha-I"].apply(self.is_high_season)
            data["min_diff"] = data.apply(self.get_min_diff, axis=1)
            data["delay"] = np.where(data["min_diff"] > self.threshold_in_minutes, 0, 1)

        shape = data.shape

        # Select relevant columns and shuffle the data
        if target_column:
            data = shuffle(
                data[["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "delay"]],
                random_state=111,
            )

        shape = data.shape

        # Generate dummy variables
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        if target_column:
            filtered_features = features[self.top_10_features]
            target_content = pd.DataFrame(data[target_column], columns=[target_column])
            return filtered_features, target_content

        if not target_column and len(features.columns) > len(self.top_10_features):
            features = features[self.top_10_features]

        return features

    def tune_hyperparameters(self):
        """
        Performs hyperparameter tuning using GridSearchCV.
        """
        data = self.load_data("data/data.csv")
        features, target = self.preprocess(data, target_column="delay")
        self.fit(features, target)
        if self.x_train is None or self.y_train is None:
            raise ValueError(
                "You must train the model first before tuning hyperparameters."
            )

        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "n_estimators": [100, 200, 300],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }

        grid_search = GridSearchCV(
            estimator=XGBClassifier(
                random_state=1,
                scale_pos_weight=self.scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            verbose=1,
            n_jobs=-1,
        )

        print("Starting hyperparameter search...")
        grid_search.fit(self.x_train, self.y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1-Score: {grid_search.best_score_}")

        # Update the model with the best parameters
        self._model = grid_search.best_estimator_
        print("Model updated with the best hyperparameters.")

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Entrena el modelo XGBoost con balanceo de clases y evalúa su desempeño.

        Args:
            features (pd.DataFrame): Características preprocesadas (top 10).
            target (pd.Series): Variable objetivo.
        """
        # Dividir los datos
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        # Calcular scale_pos_weight
        n_y0 = len(self.y_train[self.y_train == 0])
        n_y1 = len(self.y_train[self.y_train == 1])
        self.scale_pos_weight = n_y0 / n_y1
        print(f"scale_pos_weight calculado: {self.scale_pos_weight}")

        # Entrenar el modelo XGBoost con balanceo
        print("Entrenando XGBoost con balanceo de clases...")
        self._model = XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=self.scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self._model.fit(self.x_train, self.y_train)

        # Predicciones
        y_preds = self._model.predict(self.x_test)

        # Evaluar modelo
        print("Reporte de Clasificación - XGBoost con Balanceo")
        print(classification_report(self.y_test, y_preds))

        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión - XGBoost con Balanceo")
        plt.show()

        # Opcional: Guardar el modelo entrenado
        self.save_model()

    def save_model(self):
        """
        Guarda el modelo entrenado en un archivo usando joblib.
        """
        joblib.dump(self._model, self.model_path)
        print(f"Modelo guardado en {self.model_path}")

    def load_model(self):
        """
        Carga el modelo entrenado desde un archivo.
        """
        if os.path.exists(self.model_path):
            self._model = joblib.load(self.model_path)
            print(f"Modelo cargado desde {self.model_path}")
        else:
            raise FileNotFoundError(
                f"No se encontró el archivo del modelo en {self.model_path}, por favor ejecute `/tune`"
            )

    def predict(self, features: pd.DataFrame) -> list:
        """
        Realiza predicciones utilizando el modelo entrenado.

        Args:
            features (pd.DataFrame): Características preprocesadas para predecir.

        Returns:
            list: Lista de predicciones (1 para retraso, 0 para no retraso).
        """
        self.model_threshold = 0.5

        if self._model is None:
            self.load_model()

            for feature in self.top_10_features:
                if feature not in features.columns:
                    features[feature] = 0
            features = features[self.top_10_features]

            probabilities = self._model.predict_proba(features)[:, 1]
            predictions = (probabilities >= self.model_threshold).astype(int)

            return {"predict": predictions.tolist()}
