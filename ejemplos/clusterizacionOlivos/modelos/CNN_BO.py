import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

# Función para establecer la semilla
def establecer_semilla(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Función para preparar los datos
def preparar_datos(df):
    X = df.iloc[:, 3:].values
    y = df['Variedad'].apply(lambda x: 0 if x == 'PI' else 1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

    return X_train, X_test, y_train, y_test

# Definir la función objetivo para la optimización bayesiana
def cnn_optimizable(filters_1, filters_2, filters_3, filters_4, kernel_size, ff_dim, dropout_rate):
    establecer_semilla()

    # Preparar los datos
    X_train, X_test, y_train, y_test = preparar_datos(data)

    # Definir la arquitectura de la CNN
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=int(filters_1), kernel_size=int(kernel_size), activation='elu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=int(filters_2), kernel_size=int(kernel_size), activation='elu'))
    model.add(Conv1D(filters=int(filters_3), kernel_size=int(kernel_size), activation='elu'))
    model.add(Conv1D(filters=int(filters_4), kernel_size=int(kernel_size), activation='elu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(int(ff_dim), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return accuracy

data = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModeloPicual.csv')

# Aplicar optimización bayesiana
pbounds = {
    'filters_1': (10, 20),
    'filters_2': (25, 35),
    'filters_3': (60, 75),
    'filters_4': (110, 130),
    'kernel_size': (2, 5),
    'ff_dim': (32, 70),
    'dropout_rate': (0.0, 0.5)
}

optimizer = BayesianOptimization(
    f=cnn_optimizable,
    pbounds=pbounds,
    random_state=42
)

# Iniciar la optimización
optimizer.maximize(init_points=7, n_iter=20)

# Mostrar los mejores parámetros encontrados
print("Mejores parámetros:", optimizer.max)
