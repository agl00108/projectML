import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LayerNormalization
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Definir las constantes
POOL_SIZE = 2          # Mantén el tamaño del pool en 2 para que el modelo reduzca la dimensionalidad sin perder demasiada información.
BATCH_SIZE = 32        # El tamaño de lote de 32 es un valor comúnmente utilizado, pero si el modelo sigue fluctuando mucho, prueba con 64 o 16.
EPOCHS = 50            # Está bien mantener 50 épocas, aunque dependiendo del comportamiento del modelo, podrías ajustar este número.
FILTERS = 32           # Reducir el número de filtros a 32 en las primeras capas para evitar un modelo demasiado complejo en las primeras etapas.
KERNEL_SIZE = 3       # Mantén el tamaño del kernel en 3, que es una opción estándar para tareas de clasificación de series temporales o datos 1D.
DROPOUT_RATE = 0.4     # Disminuir el dropout de 0.5 a 0.4, ya que un valor más alto puede eliminar demasiado de la información útil.
FF_DIM = 64            # Reducir la dimensión de la capa densa a 64 para evitar que el modelo se haga demasiado grande y se sobreajuste.
PATIENCE = 3           # Reducir la paciencia a 3 para que el early stopping sea más sensible y detenga el entrenamiento antes si no mejora.

# Función para establecer la semilla y asegurar reproducibilidad
def establecer_semilla(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Función para mostrar los resultados
def mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test):
    print(f"Evaluación del modelo {nombre_modelo}")
    print(f"Pérdida: {loss}")
    print(f"Precisión: {accuracy}")

    # Mostrar informe de clasificación
    print("\nInforme de clasificación:")
    print(classification_report(y_true, y_pred))

    # Mostrar matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Matriz de confusión - {nombre_modelo}')
    plt.xlabel('Predicciones')
    plt.ylabel('Valores reales')
    plt.show()


def preparar_datos(df):
    # Separar las características (bandas espectrales) y el objetivo (Variedad)
    X = df.iloc[:, 3:].values  # Usar solo las bandas espectrales como características
    y = df['Variedad'].values  # La variable objetivo es la columna 'Variedad'

    # Convertir la variable objetivo a One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, categories=[['AR', 'NO AR']])
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Primeras filas de X_train:")
    print(X_train[:5])  # Imprimir las primeras 5 filas de X_train
    print("Forma de X_train:", X_train.shape)

    print("Primeras filas de y_train:")
    print(y_train[:5])  # Imprimir las primeras 5 filas de y_train
    print("Forma de y_train:", y_train.shape)

    # Redimensionar X para que sea compatible con Conv1D (se requiere un formato 3D)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, X_test, y_train, y_test


def ejecutar_cnn(df):
    establecer_semilla()
    start_time = time.time()

    # Preparar los datos
    X_train, X_test, y_train, y_test = preparar_datos(df)

    # Definir el modelo CNN
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LayerNormalization(epsilon=1e-6))
    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))

    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))

    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D(pool_size=POOL_SIZE))

    model.add(Flatten())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(FF_DIM, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(2, activation="softmax"))

    # Compilar el modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)

    # Entrenar el modelo
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.show()

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    nombre_modelo = "CNN"
    mostrar_resultados(y_true, y_pred, loss, accuracy, nombre_modelo, y_pred_prob, y_test)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Tiempo de ejecución: {execution_time:.2f} segundos')
    return history


# Cargar los datos
data = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModelo.csv')

# Ejecutar el modelo CNN con los datos
ejecutar_cnn(data)
