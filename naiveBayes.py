import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Importación de los datos
datos_final = pd.read_csv('archivos/archivosRefactorizados/linea4_21_t.csv')
print(datos_final.shape)

# 2. Dividir en predictores (X) y salida (y)
X = datos_final.drop(['Variedad'], axis=1)
y = datos_final['Variedad']

# 3. Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# 4. Binarizar los datos usando la mediana
mediana_train = np.median(X_train, axis=0)
X_train_binarizado = np.where(X_train > mediana_train, 1, 0)

mediana_test = np.median(X_test, axis=0)  # Mediana por columna para el conjunto de prueba
X_test_binarizado = np.where(X_test > mediana_test, 1, 0)

# 5. Inicializar y entrenar el modelo Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_binarizado, y_train)

# 6. Hacer predicciones en el conjunto de prueba
y_pred = bnb.predict(X_test_binarizado)

# 7. Crear un DataFrame que compare las predicciones con los valores reales
resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})

# 8. Mostrar las primeras filas del DataFrame con las predicciones y los valores reales
print(resultados.head())

# 9. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo Naive Bayes: {accuracy * 100:.2f}%')

# 10. Reporte de clasificación para ver las métricas detalladas por clase
print(classification_report(y_test, y_pred))
