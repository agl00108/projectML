import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LayerNormalization
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Definir las constantes para los parámetros del modelo
POOL_SIZE = 2
BATCH_SIZE = 32
EPOCHS = 50
FILTERS = 32
KERNEL_SIZE = 3
FF_DIM = 64
PATIENCE = 10

# Función para establecer la semilla
def establecer_semilla(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Función para mostrar los resultados del modelo
def mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test):
    print(f"Evaluación del modelo {nombre_modelo}")
    print(f"Pérdida: {loss}")
    print(f"Precisión: {accuracy}")
    print("\nInforme de clasificación:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Matriz de confusión - {nombre_modelo}')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.show()

# Función para preparar los datos
def preparar_datos(df):
    X = df.iloc[:, 3:].values
    y = df['Variedad'].apply(lambda x: 0 if x == 'AR' else 1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

    return X_train, X_test, y_train, y_test, scaler

# Función principal para ejecutar la red neuronal CNN
def ejecutar_cnn(df):
    establecer_semilla()
    start_time = time.time()

    X_train, X_test, y_train, y_test, scaler = preparar_datos(df)

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=16, kernel_size=3, activation='elu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(Conv1D(filters=32, kernel_size=3, activation='elu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='elu'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='elu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))
    model.add(Flatten())
    model.add(Dense(FF_DIM, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.show()

    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    nombre_modelo = "CNN"
    mostrar_resultados(y_test, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Tiempo de ejecución: {execution_time:.2f} segundos')

    return model, scaler  # Devolver el modelo y el scaler

# Cargar los datos desde un archivo CSV
data = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModelo.csv')

# Ejecutar el modelo CNN y obtener el modelo entrenado y el escalador
model, scaler = ejecutar_cnn(data)

# Función para preprocesar nuevos datos y realizar predicciones
def comprobar_nuevos_datos(model, nuevos_datos, scaler):
    X_nuevos = nuevos_datos.iloc[:, 3:].values
    y_nuevos = nuevos_datos['Variedad'].apply(lambda x: 0 if x == 'AR' else 1).values

    X_nuevos = X_nuevos.reshape(X_nuevos.shape[0], X_nuevos.shape[1], 1)

    X_nuevos = scaler.transform(X_nuevos.reshape(-1, X_nuevos.shape[1])).reshape(X_nuevos.shape)

    y_pred_prob = model.predict(X_nuevos)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    print("\nPredicciones para los nuevos datos:")
    print(y_pred.flatten())

    print("\nValores reales:")
    print(y_nuevos)

    print("\nInforme de clasificación para los nuevos datos:")
    print(classification_report(y_nuevos, y_pred))

    cm = confusion_matrix(y_nuevos, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Matriz de confusión - Nuevos datos')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.show()

# Cargar los nuevos datos desde un archivo CSV
nuevos_datos = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosPrueba.csv')

# Preprocesar y comprobar los nuevos datos con el modelo entrenado
comprobar_nuevos_datos(model, nuevos_datos, scaler)

