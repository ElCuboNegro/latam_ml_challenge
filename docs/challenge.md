- parte de la arquitectura no se adhiere a SOLID, el metodo `model.preprocess` no cumple con responsabilidad unica, ya que se pide que funcione con dos posibles tipos de datos.

- reentrenando los modelos los resultados son los siguientes:
XGBoost con Balanceo:
              precision    recall  f1-score   support

           0       0.88      0.53      0.66     11103
           1       0.25      0.67      0.36      2539

    accuracy                           0.56     13642
   macro avg       0.56      0.60      0.51     13642
weighted avg       0.76      0.56      0.61     13642

Regresión Logística con Balanceo:
              precision    recall  f1-score   support

           0       0.87      0.52      0.65     11103
           1       0.24      0.67      0.36      2539

    accuracy                           0.55     13642
   macro avg       0.56      0.60      0.50     13642
weighted avg       0.76      0.55      0.60     13642

Los resultados han mostrado que, para ambos modelos (XGBoost y Regresión Logística con balanceo) los resultados son muy similares, especialmente en cuanto a precisión y recall. Ambos modelos muestran un comportamiento casi idéntico en términos de:

- Precisión para la clase 1 (retrasos): 0.24-0.25 en ambos modelos, lo que indica que cuando predicen un retraso, solo el 24-25% de las veces es correcto.
- Recall para la clase 1 (retrasos): 0.67 en ambos modelos, lo que significa que están capturando aproximadamente el 67% de los vuelos que realmente se retrasan.
- F1-Score para la clase 1: 0.36, lo que muestra un equilibrio bajo entre precisión y recall para la clase de retrasos.

Recomendación:
Dado que los dos modelos presentan resultados muy similares y la Regresión Logística es más ligera computacionalmente, podría optar por Regresión Logística con balanceo, pero la evaluacion se hace en tanto al scoring.


- Cree un endpoint que fuerza el tunning de los hiperparametros del XGBOOST en `/tune`
- Cree un test para verificar que el tamaño del dataframe preprocesado sea consistente con la data.