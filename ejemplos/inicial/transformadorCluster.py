import pandas as pd

file_path = '../../archivos/archivosOriginales/cluster/linea2_20_clus.xlsx'
df = pd.read_excel(file_path)
df = df.drop(columns=['cluster_area'])
output_file = '../../archivos/archivosRefactorizados/cluster/linea2_20_clus_t.csv'
df.to_csv(output_file, index=False)

print(f'Archivo CSV generado: {output_file}')
