from model import DelayModel, ModelPersistence, CSVDataLoader
from sklearn.model_selection import train_test_split
import logging


def main():
    delay_model = DelayModel()
    data_loader = CSVDataLoader()

    # Cargar los datos
    data = data_loader.load_data("data/data.csv")

    # Preprocesar los datos
    features, target = delay_model.preprocess(data, target_column="delay")

    # Dividir los datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.33, random_state=42, stratify=target
    )

    # Entrenar el modelo con los datos de entrenamiento
    delay_model.fit(features=X_train, target=y_train)

    # Evaluar el modelo con los datos de validación
    report = delay_model.get_report(X_val, y_val)
    print("Reporte de clasificación en el conjunto de validación:")
    print(report)


if __name__ == "__main__":
    main()
