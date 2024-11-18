import pandas as pd

archivo = '../../archivos/archivosOriginales/6bandas/6Bandas.xlsx'

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
    nombre_archivo_csv = f'{nombre_rango}.csv'
    df_rango.to_csv(nombre_archivo_csv, index=False)
    print(f'Archivo creado: {nombre_archivo_csv}')
