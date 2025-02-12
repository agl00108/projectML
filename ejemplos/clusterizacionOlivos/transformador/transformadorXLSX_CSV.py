import pandas as pd
import random

# Cargar el archivo Excel
df = pd.read_excel("../../../archivos/archivosOriginales/clusterizacionOlivos/division_tres_clusters_excell_2.xlsx")

# Reemplazar 'PI_1' por 'PI'
df['Variedad'] = df['Variedad'].replace({'PI_1': 'PI'})

# Filtrar las filas donde la variedad es 'PI'
ar_rows = df[df['Variedad'] == 'PI']

# Tomar una muestra aleatoria de 30 filas para prueba
prueba_ar = ar_rows.sample(n=35, random_state=42)

# El resto será el conjunto de modelo
modelo_ar = ar_rows.drop(prueba_ar.index)
num_ar_filas = len(modelo_ar)

# Filtrar las filas donde la variedad no es 'PI' y cambiar la etiqueta a 'NO PI'
no_ar_rows = df[df['Variedad'] != 'PI'].copy()
no_ar_rows.loc[:, 'Variedad'] = 'NO PI'

# Tomar una muestra del mismo tamaño que el número de filas de 'PI' para el conjunto de modelo
modelo_no_ar = no_ar_rows.sample(n=num_ar_filas, random_state=42)

# Las filas restantes serán el conjunto de prueba
prueba_no_ar = no_ar_rows.drop(modelo_no_ar.index).sample(n=35, random_state=42)

# Combinar ambos conjuntos para crear los CSV de modelo y prueba
modelo_df = pd.concat([modelo_ar, modelo_no_ar])
modelo_df.to_csv("../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModeloPicual.csv", index=False)

prueba_df = pd.concat([prueba_ar, prueba_no_ar])
prueba_df.to_csv("../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosPruebaPicual.csv", index=False)

print("Archivos CSV generados correctamente.")
