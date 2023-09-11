import pandas as pd

# Ruta del archivo CSV
ruta_archivo = "UNION_SERVER_OMP_NO_RECURSIVO.csv"

# Leer el archivo CSV en un dataframe
df = pd.read_excel(ruta_archivo)

# Cambiar el nombre de una columna existente
df.rename(columns={'Tipo_Procesador': 'ESTRATEGIA'}, inplace=True)
df.rename(columns={'MODEL': 'ESTRATEGIA'}, inplace=True)
df.rename(columns={'RECURSIVITY': 'Recur'}, inplace=True)
df.rename(columns={'OMP_threads': 'Hilos'}, inplace=True)
df.rename(columns={'TIME': 'Time'}, inplace=True)

Tipo_Procesador	ESTRATEGIA	Tipo_Memoria	Tipo_Disco

# Guardar el dataframe modificado en un nuevo archivo CSV
ruta_salida = "UNION_SERVER_OMP_NO_RECURSIVO.csv"
df_filtrado.to_csv(ruta_salida, index=False)

print("El archivo CSV ha sido modificado exitosamente.")
