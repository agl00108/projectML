import os
import pandas as pd

def dividir_csv_por_rango(archivo_csv, columna_rango, output_dir):
    """
    Divide un archivo CSV en varios archivos según los valores únicos en una columna de rango.

    Args:
        archivo_csv (str): Ruta del archivo CSV de entrada.
        columna_rango (str): Nombre de la columna que contiene los rangos.
        output_dir (str): Ruta del directorio donde se guardarán los archivos generados.

    Returns:
        None
    """
    # Leer el archivo CSV
    df = pd.read_csv(archivo_csv)

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtener los valores únicos de la columna de rango
    rangos_unicos = df[columna_rango].unique()

    # Generar un archivo CSV por cada rango
    for rango in rangos_unicos:
        # Filtrar las filas que corresponden al rango actual
        df_rango = df[df[columna_rango] == rango]

        # Crear el nombre del archivo con base en el rango
        nombre_rango = rango.split('/')[0].strip()  # Usar solo la primera parte del rango
        nombre_archivo_csv = os.path.join(output_dir, f"{nombre_rango.replace(' ', '_')}.csv")

        # Guardar el dataframe filtrado en un archivo CSV
        df_rango.to_csv(nombre_archivo_csv, index=False)
        print(f"Archivo creado: {nombre_archivo_csv}")

# Crear el directorio si no existe
output_dir = "../../archivos/archivosRefactorizados/6bandas3-1"

os.makedirs(output_dir, exist_ok=True)

# PRIMER PASO: Filtrar y procesar los datos originales
df = pd.read_excel("../../archivos/archivosOriginales/6bandas3-1/division_olivos_3comb_1individual.xlsx")

# Eliminar las filas con rango máximo o mínimo
df_filtered = df[~df['Rango'].isin(['Rango max (Rango 3 a 10.0)', 'Rango min (0.0 a Rango -3)'])]

# Filtrar las filas donde la variedad sea "AR" y las que no son "AR"
ar_rows = df_filtered[df_filtered['Variedad'] == 'AR']
non_ar_rows = df_filtered[df_filtered['Variedad'] != 'AR']

# Seleccionar aleatoriamente el mismo número de filas que "AR"
random_non_ar_rows = non_ar_rows.sample(n=len(ar_rows), random_state=42)

# Modificar la columna 'Variedad' de las filas seleccionadas a "NO AR"
random_non_ar_rows['Variedad'] = 'NO AR'

# Concatenar las filas de "AR" con las seleccionadas de otras variedades
final_df = pd.concat([ar_rows, random_non_ar_rows])

# Ordenar el resultado
final_df = final_df.sort_index()

# SEGUNDO PASO: Guardar filas con 'Tipo' igual a "Individual" y "AR" en un CSV separado
individual_ar_rows = final_df[(final_df['Tipo'] == 'Individual') & (final_df['Variedad'] == 'AR')]
if not individual_ar_rows.empty:
    individual_ar_rows.to_csv(
        os.path.join(output_dir, "individual_ar.csv"),
        index=False
    )
    print(f'Archivo creado: {os.path.join(output_dir, "individual_ar.csv")}')

# Eliminar las filas 'Individual' y 'AR' del dataframe general
final_df = final_df[~((final_df['Tipo'] == 'Individual') & (final_df['Variedad'] == 'AR'))]

# Guardar el dataframe general en un nuevo archivo Excel
final_df.to_excel(os.path.join(output_dir, "resultado_filtrado_individual.xlsx"), index=False)

# TERCER PASO: Crear un CSV para cada uno de los rangos
archivo = os.path.join(output_dir, "resultado_filtrado_individual.xlsx")
df = pd.read_excel(archivo)

# Filtrar los valores únicos en la columna de rango
rangos_unicos = df.iloc[:, 2].unique()
for rango in rangos_unicos:
    # Filtrar filas con el rango actual
    df_rango = df[df.iloc[:, 2] == rango]

    # Renombrar el rango para usar la primera parte antes de la barra (/)
    nombre_rango = rango.split('/')[0].strip()  # Eliminar espacios y usar solo la primera parte

    # Guardar en un archivo CSV
    nombre_archivo_csv = os.path.join(output_dir, f"{nombre_rango}.csv")
    df_rango.to_csv(nombre_archivo_csv, index=False)
    print(f'Archivo creado: {nombre_archivo_csv}')

archivo="../../archivos/archivosRefactorizados/6bandas3-1/individual_ar.csv"
columna = "Rango"
directorio_salida = output_dir
#dividir_csv_por_rango(archivo, columna, directorio_salida)
