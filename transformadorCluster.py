import pandas as pd

# Ruta del archivo original
file_path = 'archivos/archivosOriginales/cluster/linea2_20_clus.xlsx'


# Leer el archivo Excel
df = pd.read_excel(file_path)

# Eliminar la columna 'cluster_area' que no es necesaria
df = df.drop(columns=['cluster_area'])

# Guardar el resultado en un archivo CSV
output_file = 'archivos/archivosRefactorizados/cluster/linea2_20_clus_t.csv'
df.to_csv(output_file, index=False)

print(f'Archivo CSV generado: {output_file}')
