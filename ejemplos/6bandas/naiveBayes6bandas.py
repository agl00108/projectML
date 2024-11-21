import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Configuración para evitar warnings relacionados con los hilos
os.environ["LOKY_MAX_CPU_COUNT"] = "16"

# Cargar los datos de los rangos
rango_media = pd.read_csv('../../archivos/archivosRefactorizados/6bandas_arbequina/Media.csv')
rango_menos1 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas_arbequina/Rango_menos1.csv')
rango_menos2 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas_arbequina/Rango_menos2.csv')
rango_1 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas_arbequina/Rango_1.csv')
rango_2 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas_arbequina/Rango_2.csv')
rango_3 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas_arbequina/Rango_3.csv')

# Lista de rangos y nombres
rangos = {
    "Media": rango_media,
    "Rango_menos1": rango_menos1,
    "Rango_menos2": rango_menos2,
    "Rango_1": rango_1,
    "Rango_2": rango_2,
    "Rango_3": rango_3
}

# Crear una lista para consolidar los resultados analíticos
resultados_analiticos = []

# Iterar sobre cada rango y aplicar el modelo
for nombre_rango, datos in rangos.items():
    print(f"Procesando: {nombre_rango}")

    # Dividir en predictores (X) y salida (y)
    X = datos.drop(['ID_OLIVO', 'Variedad', 'Rango'], axis=1)
    y = datos['Variedad']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # Binarizar los datos usando la mediana
    mediana_train = np.median(X_train, axis=0)
    X_train_binarizado = np.where(X_train > mediana_train, 1, 0)
    mediana_test = np.median(X_test, axis=0)
    X_test_binarizado = np.where(X_test > mediana_test, 1, 0)

    # Inicializar y entrenar el modelo Naive Bayes
    bnb = BernoulliNB()
    bnb.fit(X_train_binarizado, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = bnb.predict(X_test_binarizado)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo Naive Bayes para {nombre_rango}: {accuracy * 100:.2f}%')

    # Reporte de clasificación con manejo de etiquetas sin predicción
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Matriz de confusión
    matriz_confusion = confusion_matrix(y_test, y_pred)

    # Guardar métricas en la lista de resultados
    resultados_analiticos.append({
        'Rango': nombre_rango,
        'Precisión General': accuracy,
        'Precisión por Clase': {key: val['precision'] for key, val in reporte.items() if key != 'accuracy'},
        'Recall por Clase': {key: val['recall'] for key, val in reporte.items() if key != 'accuracy'},
        'F1 por Clase': {key: val['f1-score'] for key, val in reporte.items() if key != 'accuracy'},
        'Matriz de Confusión': matriz_confusion.tolist(),
        'Instancias Totales': len(y_test)
    })

# Convertir los resultados a un DataFrame y guardarlo como CSV
resultados_df = pd.json_normalize(resultados_analiticos)
resultados_df.to_csv('resultados_analiticos_naive_bayes.csv', index=False)

print("Resultados analíticos guardados en 'resultados_analiticos_naive_bayes.csv'.")

