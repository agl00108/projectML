import pandas as pd

# Cargar el archivo Excel
df = pd.read_excel("../../../archivos/archivosOriginales/clusterizacionOlivos/division_tres_clusters_excell_2.xlsx")

# Filtrar las filas donde la variedad no es 'PI'
df_filtrado = df[df['Variedad'] != 'PI'].copy()


# Guardar los CSVs con las filas filtradas
df_filtrado.to_csv("../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModeloSinPicual.csv", index=False)

print("Archivos CSV generados correctamente sin la variedad 'PI'.")
