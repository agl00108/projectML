import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Importación de los datos
datos_final = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModelo.csv')
datos_nuevos = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosPrueba.csv')

# Dividir en predictores (X), salida (y) y 'num_pixeles'
X = datos_final.drop(columns=['Variedad', 'num_pixeles'])
y = datos_final['Variedad']

X_nuevos = datos_nuevos.drop(columns=['Variedad', 'num_pixeles'])
y_nuevos = datos_nuevos['Variedad']

# Estandarizar los datos
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.25, random_state=123)

# Inicializar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo KNN
knn.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Hacer predicciones en los datos nuevos (DatosPrueba)
X_nuevos_normalizado = scaler.transform(X_nuevos)
y_pred_nuevos = knn.predict(X_nuevos_normalizado)

# Crear el contenido para el archivo .txt
with open('informe_modelo.txt', 'w') as f:
    f.write("Informe del Modelo KNN\n")
    f.write("=" * 40 + "\n")

    f.write(f"Precisión del modelo KNN: {accuracy * 100:.2f}%\n")
    f.write("\nClasificación Completa:\n")
    f.write(classification_rep)

    f.write("\nMatriz de Confusión:\n")
    f.write(str(conf_matrix))

    # Comparar las predicciones con los valores reales en el conjunto de prueba
    f.write("\nComparación de Predicciones en el Conjunto de Prueba:\n")
    resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
    f.write(resultados.to_string(index=False))

    # Predicciones en los datos nuevos
    f.write("\nPredicciones en los Datos Nuevos:\n")
    resultados_nuevos = pd.DataFrame({'Variedad Real': y_nuevos, 'Predicción': y_pred_nuevos})
    f.write(resultados_nuevos.to_string(index=False))

    f.write("\nInforme generado correctamente.\n")

print("El informe se ha generado correctamente en el archivo 'informe_modelo.txt'.")
