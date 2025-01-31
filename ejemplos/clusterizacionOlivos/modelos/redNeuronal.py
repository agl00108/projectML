import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Cargar los datos
data = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModeloArbequina.csv')

# Seleccionar las bandas espectrales como características (X) y la variedad como objetivo (y)
X = data.iloc[:, 3:]  # Usar solo las bandas espectrales como características
y = data['Variedad']

# Convertir el objetivo 'Variedad' en valores binarios (AR -> 0, NO AR -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Asignar pesos basados en el número de píxeles
sample_weights = data['num_pixeles']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X, y_encoded, sample_weights, test_size=0.3, random_state=42
)

# Sin sample weights probar
#X_train, X_test, y_train, y_test = train_test_split(  X, y_encoded, test_size=0.3, random_state=42)

# Crear y ajustar el clasificador MLP con parámetros optimizados
mlp = MLPClassifier(activation='identity',  # Función de activación tangente hiperbólica
    hidden_layer_sizes=(64,), # Cambiamos la arquitectura
    max_iter=1000,                     # Aumentamos el número de iteraciones
    learning_rate_init=0.001,          # Ajustamos la tasa de aprendizaje
    alpha=0.0001,                      # Regularización L2
    random_state=42,
    solver='adam',                     # Optimizador Adam
)
mlp.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = mlp.predict(X_test)

# Generar el informe de clasificación
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)
