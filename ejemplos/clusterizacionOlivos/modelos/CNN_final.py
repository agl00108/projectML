import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LayerNormalization
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Definir las constantes para los parámetros del modelo
POOL_SIZE = 2          # Mantén el tamaño del pool en 2 para que el modelo reduzca la dimensionalidad sin perder demasiada información.
BATCH_SIZE = 32        # El tamaño de lote de 32 es un valor comúnmente utilizado, pero si el modelo sigue fluctuando mucho, prueba con 64 o 16.
EPOCHS = 50            # Está bien mantener 50 épocas, aunque dependiendo del comportamiento del modelo, podrías ajustar este número.
FILTERS = 32           # Reducir el número de filtros a 32 en las primeras capas para evitar un modelo demasiado complejo en las primeras etapas.
KERNEL_SIZE = 3       # Mantén el tamaño del kernel en 3, que es una opción estándar para tareas de clasificación de series temporales o datos 1D.
FF_DIM = 64            # Reducir la dimensión de la capa densa a 64 para evitar que el modelo se haga demasiado grande y se sobreajuste.
PATIENCE = 5           # Early stopping sea más sensible y detenga el entrenamiento antes si no mejora.

# Función para establecer la semilla y asegurar reproducibilidad
def establecer_semilla(seed=1234):
    np.random.seed(seed)           # Establecer la semilla de NumPy para reproducibilidad
    tf.random.set_seed(seed)       # Establecer la semilla en TensorFlow
    random.seed(seed)              # Establecer la semilla de la librería random

# Función para mostrar los resultados del modelo: precisión, pérdida, clasificación y matriz de confusión
def mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test):
    print(f"Evaluación del modelo {nombre_modelo}")
    print(f"Pérdida: {loss}")
    print(f"Precisión: {accuracy}")

    # Mostrar informe de clasificación con las métricas de rendimiento
    print("\nInforme de clasificación:")
    print(classification_report(y_true, y_pred))

    # Mostrar matriz de confusión para evaluar las predicciones
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # Visualización de la matriz de confusión
    plt.title(f'Matriz de confusión - {nombre_modelo}')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.show()


# Función para preparar los datos: separar características y objetivo, y dividir en entrenamiento y prueba
def preparar_datos(df):
    # Separar las características (bandas espectrales) y el objetivo (Variedad)
    X = df.iloc[:, 3:].values  # Usar solo las columnas que contienen las bandas espectrales como características
    y = df['Variedad'].apply(lambda x: 0 if x == 'AR' else 1).values  # Convertir las categorías 'AR' -> 0 y 'NO AR' -> 1

    # Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Redimensionar X para que sea compatible con Conv1D (requiere formato 3D)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Redimensionar para 1D Conv
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Redimensionar para 1D Conv

    # Normalizar las características usando StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

    return X_train, X_test, y_train, y_test

# Función principal para ejecutar la red neuronal CNN
def ejecutar_cnn(df):
    establecer_semilla()  # Asegurar reproducibilidad
    start_time = time.time()  # Comenzar el conteo del tiempo de ejecución

    # Preparar los datos de entrada
    X_train, X_test, y_train, y_test = preparar_datos(df)

    # Definir el modelo CNN (Convolutional Neural Network)
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Capa de entrada, forma: (n_samples, n_features, 1)
    model.add(Conv1D(filters=16, kernel_size=3, activation='elu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))  # MaxPooling1D para reducir la dimensionalidad
    model.add(Conv1D(filters=32, kernel_size=3, activation='elu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='elu'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='elu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))  # MaxPooling adicional
    model.add(Flatten())  # Aplanar la salida para pasar a las capas densas
    model.add(Dense(FF_DIM, activation='relu'))  # Capa densa con 64 unidades y activación ReLU
    model.add(Dense(1, activation='sigmoid'))  # Capa de salida, activación sigmoide para clasificación binaria

    # Compilar el modelo con la función de pérdida y optimizador
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Configurar early stopping para detener el entrenamiento si no mejora después de 3 épocas
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Graficar la precisión durante las épocas
    plt.plot(history.history['accuracy'])  # Precisión del entrenamiento
    plt.plot(history.history['val_accuracy'])  # Precisión de la validación
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.show()

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test, y_test)  # Evaluar pérdida y precisión
    y_pred_prob = model.predict(X_test)  # Predecir las probabilidades
    y_pred = (y_pred_prob > 0.5).astype("int32")  # Convertir probabilidades a predicciones binarias

    # Mostrar resultados del modelo
    nombre_modelo = "CNN"
    mostrar_resultados(y_test, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test)

    end_time = time.time()  # Fin del tiempo de ejecución
    execution_time = end_time - start_time  # Calcular el tiempo de ejecución total
    print(f'Tiempo de ejecución: {execution_time:.2f} segundos')  # Mostrar el tiempo de ejecución

    return history

# Cargar los datos desde un archivo CSV
data = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModelo.csv')

# Ejecutar el modelo CNN con los datos cargados
ejecutar_cnn(data)
