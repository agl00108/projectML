import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LayerNormalization
import time

# Parámetros configurables para el modelo
POOL_SIZE = 2  # Tamaño del filtro de la capa MaxPooling
BATCH_SIZE = 32  # Tamaño del lote para el entrenamiento
EPOCHS = 100  # Número máximo de épocas
FILTERS = 64  # Número de filtros en las capas Conv1D
KERNEL_SIZE = 3  # Tamaño del kernel (filtro) en Conv1D
DROPOUT_RATE = 0.3  # Tasa de dropout para regularización
FF_DIM = 128  # Dimensión de la capa densa (fully connected)
PATIENCE = 5  # Número de épocas para early stopping antes de detener el entrenamiento si no mejora

# Función para preparar los datos de entrada
def preparar_datos(df):
    # Separar las características (X) de la variable objetivo (y)
    X = df.drop('Variedad', axis=1)  # Características
    y = df['Variedad']  # Variable objetivo

    # Escalar las características usando StandardScaler (normalización)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Codificar las etiquetas de la variable objetivo
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Reestructurar los datos para la CNN (se espera un tensor 3D: muestras, características, 1 canal)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    # Dividir los datos en conjuntos de entrenamiento (70%) y prueba (30%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

    # Convertir las etiquetas en formato one-hot encoding para clasificación multiclase
    y_train = np.eye(len(np.unique(y_encoded)))[y_train]
    y_test = np.eye(len(np.unique(y_encoded)))[y_test]

    return X_train, X_test, y_train, y_test

# Función para mostrar los resultados del modelo
def mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test):
    from sklearn.metrics import classification_report, confusion_matrix
    # Imprimir la pérdida y precisión del modelo
    print(f'Pérdida {nombre_modelo}: {loss:.4f}')
    print(f'Precisión {nombre_modelo}: {accuracy:.4f}')

    # Imprimir el reporte de clasificación que incluye precision, recall y f1-score
    print("Reporte de clasificación:")
    print(classification_report(y_true, y_pred))

    # Imprimir la matriz de confusión para visualizar errores de clasificación
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))

# Función principal que define, entrena y evalúa la CNN
def ejecutar_cnn(df):
    start_time = time.time()  # Registrar el tiempo de inicio
    X_train, X_test, y_train, y_test = preparar_datos(df)  # Preparar los datos

    # Crear un modelo secuencial de la CNN
    model = Sequential()

    # Definir la estructura de la red neuronal
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Capa de entrada
    model.add(LayerNormalization(epsilon=1e-6))  # Normalización de capas

    # Primera capa Conv1D seguida de MaxPooling
    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))

    # Segunda capa Conv1D seguida de MaxPooling
    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))

    # Tercera capa Conv1D seguida de MaxPooling
    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))

    # Aplanar las salidas de las capas convolucionales
    model.add(Flatten())

    # Añadir capa de dropout para regularización
    model.add(Dropout(DROPOUT_RATE))

    # Capa densa completamente conectada con activación ReLU
    model.add(Dense(FF_DIM, activation="relu"))

    # Añadir otra capa de dropout
    model.add(Dropout(DROPOUT_RATE))

    # Capa de salida con activación softmax (para clasificación multiclase)
    model.add(Dense(y_train.shape[1], activation="softmax"))

    # Compilar el modelo usando 'categorical_crossentropy' como función de pérdida
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Early stopping para detener el entrenamiento si la precisión en el conjunto de validación no mejora
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test, y_test)

    # Obtener las predicciones del modelo
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convertir las probabilidades en clases
    y_true = np.argmax(y_test, axis=1)  # Obtener las clases reales

    # Mostrar los resultados
    nombre_modelo = "CNN"
    mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test)

    end_time = time.time()  # Registrar el tiempo de finalización
    execution_time = end_time - start_time  # Calcular el tiempo de ejecución
    print(f'Tiempo de ejecución: {execution_time:.2f} segundos')

    return history  # Devolver el historial de entrenamiento

# Cargar el conjunto de datos desde un archivo CSV
df = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModelo.csv')

# Ejecutar la CNN usando los datos cargados
historia = ejecutar_cnn(df)
