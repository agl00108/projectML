import pandas as pd
import random

df = pd.read_excel("../../../archivos/archivosOriginales/clusterizacionOlivos/division_tres_clusters_excell.xlsx")

ar_rows = df[df['Variedad'] == 'AR']
prueba_ar = ar_rows.sample(n=5, random_state=42)
modelo_ar = ar_rows.drop(prueba_ar.index)
num_ar_filas = len(modelo_ar)

no_ar_rows = df[df['Variedad'] != 'AR'].copy()
no_ar_rows.loc[:, 'Variedad'] = 'NO AR'
modelo_no_ar = no_ar_rows.sample(n=num_ar_filas, random_state=42)
prueba_no_ar = no_ar_rows.drop(modelo_no_ar.index).sample(n=5, random_state=42)

modelo_df = pd.concat([modelo_ar, modelo_no_ar])
modelo_df.to_csv("../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosModelo.csv", index=False)

prueba_df = pd.concat([prueba_ar, prueba_no_ar])
prueba_df.to_csv("../../../archivos/archivosRefactorizados/clusterizacionOlivos/DatosPrueba.csv", index=False)

print("Archivos CSV generados correctamente.")

