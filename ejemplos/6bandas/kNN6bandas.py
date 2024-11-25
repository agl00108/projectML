import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "16"

# Cargar los datos de los rangos
rango_media = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-individual/Media.csv')
rango_menos1 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-individual/Rango_menos1.csv')
rango_menos2 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-individual/Rango_menos2.csv')
rango_1 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-individual/Rango_1.csv')
rango_2 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-individual/Rango_2.csv')
rango_3 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-individual/Rango_3.csv')

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

    # Estandarizar los datos
    scaler = StandardScaler()
    X_normalizado = scaler.fit_transform(X)

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.25, random_state=123)

    # Inicializar y entrenar el modelo KNN
    knn = KNeighborsClassifier(metric='euclidean')
    knn.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = knn.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo KNN para {nombre_rango}: {accuracy * 100:.2f}%')

    # Reporte de clasificación
    reporte = classification_report(y_test, y_pred, output_dict=True)

    # Matriz de confusión
    matriz_confusion = confusion_matrix(y_test, y_pred)

    # Preparar los datos analíticos del modelo
    resultados_analiticos.append({
        'Rango': nombre_rango,
        'Precisión General': accuracy,
        'Precisión por Clase': {key: val['precision'] for key, val in reporte.items() if key != 'accuracy'},
        'Recall por Clase': {key: val['recall'] for key, val in reporte.items() if key != 'accuracy'},
        'F1 por Clase': {key: val['f1-score'] for key, val in reporte.items() if key != 'accuracy'},
        'Matriz de Confusión': matriz_confusion.tolist(),  # Convertir matriz a lista para CSV
        'Instancias Totales': len(y_test)
    })

# Convertir los resultados a un DataFrame y guardarlo como CSV
resultados_df = pd.json_normalize(resultados_analiticos)
resultados_df.to_csv('resultados_analiticos_kNN_individual.csv', index=False)

print("Resultados analíticos guardados en 'resultados_analiticos_kNN_individual.csv'.")