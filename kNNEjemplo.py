import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

#Importación de los datos
datos_final = pd.read_csv('./archivos/resultado/linea2_corr_20.csv')
print(datos_final.shape)
#print(datos_final.head())

# Dividir en predictores (X) y salida (y)
X = datos_final.drop(['Variedad'], axis=1)
y = datos_final['Variedad']

# 3. Estandarizar los datos
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# 4. Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.25, random_state=123)

# 5. Inicializar y entrenar el modelo KNN
knn = KNeighborsClassifier(metric='euclidean')
knn.fit(X_train, y_train)

# 6. Hacer predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# 7. Crear un DataFrame que compare las predicciones con los valores reales
resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})

# 8. Mostrar las primeras filas del DataFrame con las predicciones y los valores reales
print(resultados.head())

# 7. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo KNN: {accuracy * 100:.2f}%')

# 8. Reporte de clasificación para ver las métricas detalladas por clase
print(classification_report(y_test, y_pred))
