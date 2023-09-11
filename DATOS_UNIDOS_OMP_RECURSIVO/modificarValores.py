import pandas as pd

# Cargar el DataFrame desde un archivo CSV (ejemplo)
# Leer los archivos CSV y crear los DataFrames
df1 = pd.read_csv('UNION_SERVER_OMP_RECURSIVO.csv')
df2 = pd.read_csv('DATOS_OPENMP_LAPTOP_OMP_RECURSIVO.csv')

# Concatenar los DataFrames
df_concatenado = pd.concat([df1, df2], ignore_index=True)


df_concatenado.to_csv('UNION_FINAL/UNION_OMP_RECURSIVO.csv', index=False)

# Mostrar el DataFrame actualizado
#print(df)
