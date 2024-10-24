import pandas as pd

file_path = 'archivos/archivosOriginales/olivo/linea4_21.xlsx'
df = pd.read_excel(file_path)

# Crear un nuevo DataFrame con las columnas deseadas
olivo_grouped = df.pivot_table(index=['ID_OLIVO', 'Variedad'],
                               columns='cluster_class',
                               values='proporción',
                               aggfunc='sum')

olivo_grouped.columns = [f'cluster_{int(col)}' for col in olivo_grouped.columns]

# Por archivo hay que ver si algunas filas sobran
if 'cluster_21' in olivo_grouped.columns:
   olivo_grouped = olivo_grouped.drop(columns=['cluster_21'])

#if 'cluster_11' in olivo_grouped.columns:
#    olivo_grouped = olivo_grouped.drop(columns=['cluster_11'])

olivo_grouped.reset_index(inplace=True)
output_file = 'archivos/archivosRefactorizados/olivo/linea4_21_t.csv'
olivo_grouped.to_csv(output_file, index=False)

print(f'Archivo CSV generado: {output_file}')

