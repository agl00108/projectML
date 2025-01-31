import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar los datos
datos_final = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModeloArbequina.csv')
datos_nuevos = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosPruebaArbequina.csv')

# Separar características y la variable objetivo
X = datos_final.drop(['IDENTIFICADOR','Variedad', 'num_pixeles'], axis=1)
y = datos_final['Variedad']

X_nuevos = datos_nuevos.drop(columns=['IDENTIFICADOR','Variedad', 'num_pixeles'])
y_nuevos = datos_nuevos['Variedad']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Binarizar los datos usando la mediana de las columnas
mediana_train = X_train.median()
X_train_binarizado = (X_train > mediana_train).astype(int)
X_test_binarizado = (X_test > mediana_train).astype(int)

# Inicializar y entrenar el modelo Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_binarizado, y_train)

# Predecir el conjunto de prueba
y_pred = bnb.predict(X_test_binarizado)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Binarizar los nuevos datos
X_nuevos_binarizado = (X_nuevos > mediana_train).astype(int)
y_pred_nuevos = bnb.predict(X_nuevos_binarizado)

# Generar el informe del modelo
with open('informes/informe_modelo_naive_bayes.txt', 'w') as f:
    f.write("Informe del Modelo Naive Bayes\n")
    f.write("=" * 40 + "\n")

    f.write(f"Precisión del modelo Naive Bayes: {accuracy * 100:.2f}%\n")
    f.write("\nClasificación Completa:\n")
    f.write(classification_rep)

    f.write("\nMatriz de Confusión:\n")
    f.write(str(conf_matrix))

    # Comparar las predicciones con los valores reales en el conjunto de prueba
    f.write("\nComparación de Predicciones en el Conjunto de Prueba:\n")
    resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
    f.write(resultados.to_string(index=False))

    f.write("\nPredicciones en los Datos Nuevos:\n")
    resultados_nuevos = pd.DataFrame({'Variedad Real': y_nuevos, 'Predicción': y_pred_nuevos})
    f.write(resultados_nuevos.to_string(index=False))

    f.write("\nInforme generado correctamente.\n")

print("Informe generado correctamente: 'informe_modelo_naive_bayes.txt'")
