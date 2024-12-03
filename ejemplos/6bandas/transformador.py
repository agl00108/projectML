import pandas as pd
import numpy as np

#PRIMER PASO: Con el archivo original, quitamos los rangos máximo y mínimo y obtenemos solo
#las columnas de ARBEQUINA y de las demás nos quedamos con el mismo número de filas
#pero aleatorio y las clasificamos como NO AR

df = pd.read_excel("../../archivos/archivosOriginales/6bandas-juntos/olivoIndividual.xlsx")

#Eliminar las filas con rango max o rango min "
df_filtered = df[~df['Rango'].isin(['Rango max (Rango 3 a 10.0)', 'Rango min (0.0 a Rango -3)'])]

#Filtrar las filas donde la variedad sea "AR" y las que no son "AR"
ar_rows = df_filtered[df_filtered['Variedad'] == 'AR']
non_ar_rows = df_filtered[df_filtered['Variedad'] != 'AR']

# Verificar si hay suficientes filas en non_ar_rows
if len(non_ar_rows) >= len(ar_rows):
    # Seleccionar aleatoriamente el mismo número de filas que "AR"
    random_non_ar_rows = non_ar_rows.sample(n=len(ar_rows), random_state=42)
    # Modificar la columna 'Variedad' de las filas seleccionadas a "NO AR"
    random_non_ar_rows['Variedad'] = 'NO AR'
else:
    print(f"No hay suficientes filas en 'non_ar_rows'. Tiene {len(non_ar_rows)} filas y se necesitan {len(ar_rows)}.")
    # Si no hay suficientes filas, tomar todas las filas de non_ar_rows
    random_non_ar_rows = non_ar_rows
    random_non_ar_rows['Variedad'] = 'NO AR'

# Concatenar las filas de "AR" con las seleccionadas de otras variedades
final_df = pd.concat([ar_rows, random_non_ar_rows])

# Ordenar el resultado
final_df = final_df.sort_index()

# Guardar el resultado en un nuevo archivo Excel
final_df.to_excel("../../archivos/archivosRefactorizados/6bandas-juntos/resultado_filtrado_individual.xlsx", index=False)

#SEGUNDO PASO: a partir del excel creado, creamos un csv para cada uno de los rangos
archivo = '../../archivos/archivosRefactorizados/6bandas-juntos/resultado_filtrado_individual.xlsx'
df = pd.read_excel(archivo)

# Filtrar los valores únicos en la columna de rango (ejemplo: "Rango 3 / Rango 2")
rangos_unicos = df.iloc[:, 2].unique()

# Crear un CSV para cada rango único
for rango in rangos_unicos:
    # Filtrar filas con el rango actual
    df_rango = df[df.iloc[:, 2] == rango]

    # Renombrar el rango para usar la primera parte antes de la barra (/)
    nombre_rango = rango.split('/')[0].strip()  # Eliminar espacios y usar solo la primera parte

    # Guardar en un archivo CSV
    nombre_archivo_csv = f'../../archivos/archivosRefactorizados/6bandas-juntos/individual/{nombre_rango}.csv'
    df_rango.to_csv(nombre_archivo_csv, index=False)
    print(f'Archivo creado: {nombre_archivo_csv}')

