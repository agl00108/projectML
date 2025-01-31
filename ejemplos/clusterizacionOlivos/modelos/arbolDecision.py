import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

datos_final = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModeloArbequina.csv')
datos_nuevos = pd.read_csv('../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosPruebaArbequina.csv')

X = datos_final.drop(columns=['IDENTIFICADOR','Variedad'])
y = datos_final['Variedad']
num_pixeles = datos_final['num_pixeles']

X_nuevos = datos_nuevos.drop(columns=['IDENTIFICADOR','Variedad'])
y_nuevos = datos_nuevos['Variedad']

scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

X_train, X_test, y_train, y_test, num_pixeles_train, num_pixeles_test = train_test_split(X_normalizado, y, num_pixeles, test_size=0.25, random_state=123)
# Asignar peso a las muestras basado en el número de píxeles (puedes hacer que el peso sea proporcional a 'num_pixeles')
sample_weight = num_pixeles_train

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train, sample_weight=sample_weight)
y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo Árbol de Decisión: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
X_nuevos_normalizado = scaler.transform(X_nuevos)
y_pred_nuevos = tree.predict(X_nuevos_normalizado)

resultados_nuevos = pd.DataFrame({'Variedad Real': y_nuevos, 'Predicción': y_pred_nuevos})


with open('informes/informe_arbol.txt', 'w') as f:
    f.write(f"Precisión del modelo Árbol de Decisión: {accuracy * 100:.2f}%\n")
    f.write("\nClasificación Completa:\n")
    f.write(classification_report(y_test, y_pred))

    f.write("\nComparación de Predicciones en el Conjunto de Prueba:\n")
    f.write(resultados.to_string(index=False))

    f.write("\nPredicciones en los Datos Nuevos:\n")
    f.write(resultados_nuevos.to_string(index=False))

    f.write("\nInforme generado correctamente.\n")

print("Informe de Árbol de Decisión generado en 'informe_arbol.txt'.")
