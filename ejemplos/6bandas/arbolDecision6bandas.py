import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Configuración para evitar warnings relacionados con los hilos
os.environ["LOKY_MAX_CPU_COUNT"] = "16"

# Cargar los datos de los rangos
rango_media = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-juntos/olivos/Media.csv')
rango_menos1 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-juntos/olivos/Rango_menos1.csv')
rango_menos2 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-juntos/olivos/Rango_menos2.csv')
rango_1 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-juntos/olivos/Rango_1.csv')
rango_2 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-juntos/olivos/Rango_2.csv')
rango_3 = pd.read_csv('../../archivos/archivosRefactorizados/6bandas-juntos/olivos/Rango_3.csv')

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

    # Inicializar el modelo con el criterio de entropía
    dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)

    # Entrenar el modelo
    dtc.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = dtc.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo Árbol de Decisión para {nombre_rango}: {accuracy * 100:.2f}%')

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
resultados_df.to_csv('resultados_analiticos_arbol_decision_individual.csv', index=False)

print("Resultados analíticos guardados en 'resultados_analiticos_arbol_decision_individual.csv'.")

#-------------------------------------------- PREDICCIÓN --------------------------------------------
# Crear un diccionario para almacenar las predicciones por rango
predicciones_consolidadas = []

# Lista de archivos CSV con los nuevos olivos por rango
archivos_nuevos = {
    "Media": '../../archivos/archivosRefactorizados/6bandas-juntos/individual/Media.csv',
    "Rango_menos1": '../../archivos/archivosRefactorizados/6bandas-juntos/individual/Rango_menos1.csv',
    "Rango_menos2": '../../archivos/archivosRefactorizados/6bandas-juntos/individual/Rango_menos2.csv',
    "Rango_1": '../../archivos/archivosRefactorizados/6bandas-juntos/individual/Rango_1.csv',
    "Rango_2": '../../archivos/archivosRefactorizados/6bandas-juntos/individual/Rango_2.csv',
    "Rango_3": '../../archivos/archivosRefactorizados/6bandas-juntos/individual/Rango_3.csv'
}

# Iterar sobre cada archivo y rango
for nombre_rango, archivo in archivos_nuevos.items():
    print(f"Procesando nuevos olivos para: {nombre_rango}")

    # Cargar los datos nuevos
    nuevos_olivos = pd.read_csv(archivo)

    # Guardar la columna 'Variedad' antes de eliminarla
    variedad_original = nuevos_olivos['Variedad']

    # Preprocesar los datos nuevos (eliminar columnas innecesarias)
    X_nuevos = nuevos_olivos.drop(['ID_OLIVO', 'Variedad', 'Rango'], axis=1)

    # Realizar predicciones para los nuevos olivos
    y_nuevos_pred = dtc.predict(X_nuevos)

    # Crear un DataFrame temporal con las predicciones y las variedades originales
    df_predicciones = pd.DataFrame({
        'ID_OLIVO': nuevos_olivos['ID_OLIVO'],
        'Rango': nombre_rango,
        'Predicción': y_nuevos_pred,
        'Variedad_Original': variedad_original
    })

    # Añadir al DataFrame consolidado
    predicciones_consolidadas.append(df_predicciones)

# Concatenar todas las predicciones en un único DataFrame
predicciones_consolidadas_df = pd.concat(predicciones_consolidadas, ignore_index=True)

# Guardar todas las predicciones en un archivo CSV general
predicciones_consolidadas_df.to_csv('predicciones_con_variedad_arbol_decision.csv', index=False)

print("Todas las predicciones y variedades originales guardadas en 'predicciones_con_variedad_arbol_decision.csv'.")
