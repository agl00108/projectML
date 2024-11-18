import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Importación de los datos, cada una de las 6 bandas
rango_media = pd.read_csv('../../archivos/archivosOriginales/aplicacionBandasBinaria/divisionRangos/Media.csv')
rango_menos1 = pd.read_csv('../../archivos/archivosOriginales/aplicacionBandasBinaria/divisionRangos/Rango_menos1.csv')
rango_menos2 = pd.read_csv('../../archivos/archivosOriginales/aplicacionBandasBinaria/divisionRangos/Rango_menos2.csv')
rango_1 = pd.read_csv('../../archivos/archivosOriginales/aplicacionBandasBinaria/divisionRangos/Rango_1.csv')
rango_2 = pd.read_csv('../../archivos/archivosOriginales/aplicacionBandasBinaria/divisionRangos/Rango_2.csv')
rango_3 = pd.read_csv('../../archivos/archivosOriginales/aplicacionBandasBinaria/divisionRangos/Rango_3.csv')

#-----------------------APLICACIÓN DEL MÉTODO PARA EL RANGO MEDIO----------------------------------------------------
# Dividir en predictores (X), todos menos la variedad, id_olivo y rango y salida (Y-Variedad)
X = rango_media.drop(['ID_OLIVO', 'Variedad', 'Rango'], axis=1)
y = rango_media['Variedad']
#Para verificar cuantas hay de cada uno, si están balanceados estaría bien
#print(rango_media.Variedad.value_counts())

'''
Estandarizamos los datos con StandardScaler(), estandarizar las características eliminando la media y 
escalando a la varianza unitaria.
'''
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