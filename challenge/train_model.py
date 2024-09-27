from challenge.model import DelayModel


def main():
    delay_model = DelayModel()

    # Cargar los datos
    data = delay_model.load_data("../data/data.csv")
    # Preprocesar los datos
    features, target = delay_model.preprocess(data, target_column="delay")

    # Entrenar el modelo con XGBoost balanceado
    delay_model.fit(features, target)

    # Opcional: Ajustar hiperparámetros
    # delay_model.tune_hyperparameters()


if __name__ == "__main__":
    main()
