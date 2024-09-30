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

- Se ha creado un script para buscar los mejores hiperparámetros del modelo XGBoost en `challenge/model.py - model.tune_hyperparameters`. Los resultados obtenidos son:

  - **Mejores parámetros:** `{'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 500, 'subsample': 0.5}`
  - **Mejor puntuación F1 clase 0:** 0.8976576077326752

- Uno de los test unitarios no tenia sentido, ya que evaluaba que el recall sea menor a 0.60, cuando en realidad se buscaba que fuera mayor. Generalmente en problemas de clasificacion, buscamos que el recall y el F1 Score sean lo mas altas posibles, mientras que el test reflejaba que se buscaba un rendimiento minimo.

---

## API

- Se está utilizando un sistema de validación de datos basado en Pydantic.
- Se realiza la validación del campo `OPERA` para asegurar que el valor concuerde con los utilizados en el entrenamiento del modelo. Aunque no todos los valores se encuentran dentro de los features utilizados, se considera un valor de entrada válido.

---

## Despliegue

### Descripción de Pasos en CI

- Activación del Workflow:
  - Se activa en cada push a las ramas main, develop y cualquier rama que coincida con feature/*.

- Pasos del Job build-and-test:
  - Paso 1: Checkout del repositorio
    - Se utiliza actions/checkout@v3 para obtener el código fuente del repositorio.
  - Paso 2: Cache de dependencias pip
    - Se implementa una caché para las dependencias instaladas mediante pip, optimizando el tiempo de instalación en ejecuciones futuras.
  - Paso 3: Configurar Python 3.9
    - Se establece el entorno de Python necesario para el proyecto.
  - Paso 4: Instalar dependencias de pruebas
    - Se instalan las dependencias necesarias tanto para la aplicación como para las pruebas.
  - Paso 5: Ejecutar pruebas del modelo
    - Se ejecutan los tests unitarios y de integración relacionados con el modelo, generando reportes de cobertura.
  - Paso 6: Ejecutar pruebas de la API
    - Se ejecutan los tests de la API, incluyendo reportes de cobertura y resultados en formato XML para integraciones adicionales si fuera necesario.
  - Paso 7: Subir reportes de pruebas (opcional)
    - Se suben los reportes generados como artefactos del workflow, permitiendo su análisis posterior.

### Descripción de Pasos en CD
#### Activación del Workflow:
CD para PRs: Se activa cuando se crea o actualiza una Pull Request hacia la rama main.
CD para Producción: Se activa en cada push a la rama main, es decir, cuando una PR es mergeada exitosamente.
#### Pasos Comunes en CD:
Checkout del repositorio
Configuración de Google Cloud SDK
  - Utiliza google-github-actions/setup-gcloud@v1 para configurar las credenciales y el proyecto de GCP.
  - Configuración de Docker para Artifact Registry
    - Prepara Docker para autenticarse y subir imágenes al registro de contenedores de Google.

#### Construcción y Push de la imagen Docker
  - Construye la imagen Docker de la aplicación y la sube al Artifact Registry de GCP.

#### Pasos Específicos en CD para PRs en main:
Despliegue en Cloud Run (Ambiente de Pruebas)
  - Despliega la aplicación en un servicio de Cloud Run nombrado según el número de la PR (latam-ml-challenge-pr-<número_de_pr>).
  - Obtención de la URL del servicio desplegado
    - Almacena la URL del servicio para ser utilizada en las pruebas de estrés.
  - Instalación de dependencias para Stress Testing
    - Instala locust y otras dependencias necesarias para ejecutar las pruebas.
  - Ejecución de Stress Tests
    - Se ejecutan pruebas de estrés utilizando locust, simulando múltiples usuarios y peticiones para verificar el desempeño de la aplicación bajo carga.
  - Destrucción del servicio de prueba
    - Independientemente del resultado de las pruebas, se elimina el servicio desplegado para liberar recursos.

#### Pasos Específicos en CD para Producción:
  - Despliegue en Cloud Run (Producción)
    - Despliega la aplicación en el servicio de producción (latam-ml-challenge), haciéndola disponible para los usuarios finales.
