import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Importación de los datos
datos_final = pd.read_csv('./archivos/resultado/linea2_corr_20.csv')
print(datos_final.shape)

# 2. Dividir en predictores (X) y salida (y)
X = datos_final.drop(['Variedad'], axis=1)
y = datos_final['Variedad']

# 3. Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# 4. Inicializar el modelo con el criterio de entropía
dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)

# 5. Entrenar el modelo
dtc.fit(X_train, y_train)

# 6. Verificar la importancia de cada atributo (entropía)
importancias = dtc.feature_importances_
print("Importancias de cada atributo:", importancias)

# 7. Hacer predicciones en el conjunto de prueba
y_pred = dtc.predict(X_test)

# 8. Crear un DataFrame que compare las predicciones con los valores reales
resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})

# 9. Mostrar las primeras filas del DataFrame con las predicciones y los valores reales
print(resultados.head())

# 10. Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo Árbol de Decisión: {accuracy * 100:.2f}%')

# 11. Reporte de clasificación para ver las métricas
