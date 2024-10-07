import pandas as pd

file_path = './archivos/linea2_corr_20.xlsx'
df = pd.read_excel(file_path)

# Crear un nuevo DataFrame con las columnas deseadas
olivo_grouped = df.pivot_table(index=['ID_OLIVO', 'Variedad'],
                               columns='cluster_class',
                               values='proporción',
                               aggfunc='sum')

# Renombrar columnas para que tengan el formato "cluster_X"
olivo_grouped.columns = [f'cluster_{int(col)}' for col in olivo_grouped.columns]

# Eliminar la columna 'cluster_11' si existe
#if 'cluster_11' in olivo_grouped.columns:
#   olivo_grouped = olivo_grouped.drop(columns=['cluster_11'])

# Restablecer el índice para que se vea como un DataFrame normal
olivo_grouped.reset_index(inplace=True)

# Guardar el resultado en un archivo CSV
output_file = './archivos/resultado/linea2_corr_20.csv'
olivo_grouped.to_csv(output_file, index=False)

print(f'Archivo CSV generado: {output_file}')

