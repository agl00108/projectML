import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Importación de los datos
datos_final = pd.read_csv('archivos/archivosRefactorizados/olivo/linea4_21_t.csv')
print(datos_final.shape)

# Dividir en predictores (X) y salida (y)
X = datos_final.drop(['Variedad'], axis=1)
y = datos_final['Variedad']

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
print(f'Precisión del modelo KNN: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))


# Predicción de nuevos olivos

# Función para predecir la variedad mayoritaria de un nuevo olivo
def predecir_variedad_mayoritaria(nuevo_olivo_df):
    # Estandarizar los datos del nuevo olivo
    nuevo_olivo_normalizado = scaler.transform(nuevo_olivo_df.drop(['Variedad'], axis=1))

    # Predecir la variedad de cada clúster del nuevo olivo
    predicciones = knn.predict(nuevo_olivo_normalizado)

    # Contar las ocurrencias de cada variedad y devolver la mayoritaria
    variedad_mayoritaria = Counter(predicciones).most_common(1)[0][0]

    return variedad_mayoritaria


# Ejemplo con nuevos datos de olivos
nuevos_olivos_df = pd.read_csv('archivos/nuevos_olivos.csv')  # Supongamos que tienes un archivo con los nuevos olivos
nuevos_olivos_df['Variedad_Predicha'] = nuevos_olivos_df.groupby('ID_OLIVO').apply(predecir_variedad_mayoritaria)

print(nuevos_olivos_df[['ID_OLIVO', 'Variedad_Predicha']].drop_duplicates())
