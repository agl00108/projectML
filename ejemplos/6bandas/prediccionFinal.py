import pandas as pd

# Cargar los datos desde el CSV
df = pd.read_csv('../6bandas/resultados/resultados3-1/kNN/predicciones_con_variedad_kNN.csv')

# Función para calcular la predicción final
def calcular_prediccion_final(predicciones):
    conteo = predicciones.value_counts()
    if len(conteo) == 1:
        return conteo.index[0]  # Solo hay un valor
    elif conteo.iloc[0] > conteo.iloc[1]:
        return conteo.index[0]  # Mayoría
    else:
        return "Empate"  # Hay empate

# Agrupar por olivo y calcular la predicción final
resultados = df.groupby('ID_OLIVO').agg({
    'Predicción': calcular_prediccion_final,
    'Variedad_Original': 'first'  # Tomamos la primera porque es igual para todo el olivo
}).reset_index()

# Renombrar las columnas para indicar que son los resultados finales
resultados.rename(columns={'Predicción': 'Predicción_Final'}, inplace=True)

# Crear un DataFrame con las columnas en el mismo formato que el original para poder concatenar
resultados['Rango'] = 'Final'  # Añadir la etiqueta "Final" en la columna Rango
resultados = resultados[['ID_OLIVO', 'Rango', 'Predicción_Final', 'Variedad_Original']]

# Renombrar la columna 'Predicción_Final' a 'Predicción' para que coincida con el CSV original
resultados.rename(columns={'Predicción_Final': 'Predicción'}, inplace=True)

# Crear una fila para "RESUMEN DE RESULTADOS"
resumen_fila = pd.DataFrame([['RESUMEN DE RESULTADOS', '', '', '']], columns=df.columns)

# Concatenar la fila "RESUMEN DE RESULTADOS" y los resultados al DataFrame original
df_final = pd.concat([df, resumen_fila, resultados], ignore_index=True)

# Guardar los resultados en el mismo archivo CSV
df_final.to_csv('../6bandas/resultados/resultados3-1/kNN/predicciones_con_variedad_kNN.csv', index=False)

print("Predicciones finales añadidas al CSV original.")