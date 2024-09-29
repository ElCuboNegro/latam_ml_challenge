# Proyecto Notebook

## Tabla de Contenidos

1. [Arquitectura](#arquitectura)
2. [Debug](#debug)
3. [Modelo](#modelo)
4. [API](#api)
5. [Despliegue](#despliegue)

---

## Arquitectura

- Parte de la arquitectura no se adhiere a los principios SOLID. El método `model.preprocess` no cumple con la responsabilidad única, ya que está diseñado para funcionar con dos tipos de datos diferentes: limpieza del dataset y entrada de datos al modelo para predicción.

---

## Debug

- **Endpoints y Pruebas:**
  - Se ha creado un endpoint que fuerza el ajuste de los hiperparámetros del modelo XGBoost en `/tune`.
  - Se ha implementado una prueba para verificar que el tamaño del dataframe preprocesado sea consistente con los datos originales en `tests/model/test_model.py` (fue necesaria para hacer verificaciones mientras desarrollaba la lógica del preprocesamiento).

---

## Modelo

### Reentrenamiento de Modelos

**XGBoost con Balanceo**

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| 0     | 0.88      | 0.53   | 0.66     | 11,103  |
| 1     | 0.25      | 0.67   | 0.36     | 2,539   |

| Métrica                        | Valor |
|--------------------------------|-------|
| Exactitud                      | 0.56  |
| Promedio Macro                 | 0.56  |
| Promedio Macro (Recall)        | 0.60  |
| Promedio Macro (F1-Score)      | 0.51  |
| Promedio Ponderado             | 0.76  |
| Promedio Ponderado (Recall)    | 0.56  |
| Promedio Ponderado (F1-Score)  | 0.61  |
| Soporte Total                  | 13,642|

**Regresión Logística con Balanceo**

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| 0     | 0.87      | 0.52   | 0.65     | 11,103  |
| 1     | 0.24      | 0.67   | 0.36     | 2,539   |

| Métrica                        | Valor |
|--------------------------------|-------|
| Exactitud                      | 0.55  |
| Promedio Macro                 | 0.56  |
| Promedio Macro (Recall)        | 0.60  |
| Promedio Macro (F1-Score)      | 0.50  |
| Promedio Ponderado             | 0.76  |
| Promedio Ponderado (Recall)    | 0.55  |
| Promedio Ponderado (F1-Score)  | 0.60  |
| Soporte Total                  | 13,642|

### Análisis de Resultados

Para ambos modelos (XGBoost y Regresión Logística con balanceo) los resultados son muy similares, especialmente en términos de precisión y recall:

- **Precisión para la clase 1 (retrasos):** 0.24-0.25 en ambos modelos, lo que indica que cuando se predice un retraso, solo el 24-25% de las veces es correcto.
- **Recall para la clase 1 (retrasos):** 0.67 en ambos modelos, lo que significa que se están capturando aproximadamente el 67% de los vuelos que realmente se retrasan.
- **F1-Score para la clase 1:** 0.36, mostrando un equilibrio bajo entre precisión y recall para la clase de retrasos.

### Clasificación

- El dataset está muy desbalanceado. Por ello, el parámetro `scale_pos_weight` se calcula como la raíz cuadrada de la proporción de muestras de la clase mayoritaria respecto a la clase minoritaria. Esto limita el efecto de una multiplicación excesiva de ejemplos positivos por pesos demasiado altos.
  
- Se ha creado un script para buscar los mejores hiperparámetros del modelo XGBoost en `challenge/model.py - model.tune_hyperparameters`. Los resultados obtenidos son:

  - **Mejores parámetros:** `{'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 500, 'subsample': 0.5}`
  - **Mejor puntuación F1:** 0.8976576077326752

- Uno de los test unitarios no tenia sentido, ya que evaluaba que el recall sea menor a 0.60, cuando en realidad se buscaba que fuera mayor. Generalmente en problemas de clasificacion, buscamos que el recall y el F1 Score sean lo mas altas posibles, mientras que el test reflejaba que se buscaba un rendimiento minimo.

---

## API

- Se está utilizando un sistema de validación de datos basado en Pydantic.
- Se realiza la validación del campo `OPERA` para asegurar que el valor concuerde con los utilizados en el entrenamiento del modelo. Aunque no todos los valores se encuentran dentro de los features utilizados, se considera un valor de entrada válido.

---

## Despliegue

- *(No se proporcionaron notas específicas sobre despliegue. Asegúrese de agregar información relevante según el proyecto.)*