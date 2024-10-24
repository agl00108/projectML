import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Importación de los datos
datos_final = pd.read_csv('archivos/archivosRefactorizados/cluster/linea4_11_clus_t.csv')

# Dividir en predictores (X) y salida (y)
X = datos_final[['cluster', 'proporcion']]
y = datos_final['Variedad']
print("DATOS")
print(datos_final.head())
#-------------------------------------------
smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y) #Regenera una nueva muestra

#Unión de los datos balanceados
datos_final = pd.concat([X, y], axis=1)

#Verificación 2 - balanceamiento
ax = sns.countplot(x='Variedad', data=datos_final)

#print(datos_final.Variedad.value_counts())
#plt.show()
#------------------------
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

# 7. Crear un DataFrame que compare las predicciones con los valores reales
resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})

# 8. Mostrar las primeras filas del DataFrame con las predicciones y los valores reales
print(resultados.head())

resultados.to_csv('predicciones_resultados.csv', index=False)