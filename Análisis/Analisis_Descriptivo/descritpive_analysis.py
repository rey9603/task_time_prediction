import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import matplotlib

matplotlib.use('Agg')

def Distribucion(output_folder,ruta,columnas_analysis):
		#Crear la carpeta de Gráficas

	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

	
	#Cargar el dataframe
	df = pd.read_csv(ruta)

	#Eliminar los valores faltantes
	df=df.dropna()

	# Configura una figura antes del bucle
	fig = plt.figure()


	for column in columnas_analysis:
	    sns.histplot(df[column], kde=True)
	    plt.xlabel('Valor')
	    plt.ylabel('Frecuencia')
	    plt.title(f'Distribución de {column}')
	    #plt.show()
	    # Genera el nombre del archivo de salida
	    filename = f'{output_folder}/{column}_histograma.png'
	    
	    # Guarda la figura en el archivo
	    plt.savefig(filename)
	    
	    # Limpia la figura actual para la siguiente iteración
	    plt.clf()

	#Cerrar la figura
	plt.close(fig)

def CorrelationPearson(output_folder,ruta,columnas_analysis):
	
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

	#Cargar el dataframe
	df = pd.read_csv(ruta)

	#Eliminar los valores faltantes
	df=df.dropna()
	#Filtrar las columnas a analizar
	df_correlation=df.loc[:,columnas_analysis]

	# Calcula la matriz de correlación y el p-valor utilizando el coeficiente de Pearson
	corr_matrix = df_correlation.corr(method='pearson')

	# Configura una figura antes del bucle
	fig = plt.figure()


	# Imprime la matriz de correlación
	print()
	print(corr_matrix)

	# Opcional: Puedes guardar la matriz de correlación en un archivo CSV
	corr_matrix.to_csv(f'{output_folder}/pearson.csv', index=True)
	#filename = f'{output_folder}/MatrixCorrelacion.png'
	    
	# Genera el mapa de calor de la matriz de correlación
	heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)

	# Rotación de los nombres de las columnas
	heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=20, horizontalalignment='right')

	fig = plt.gcf()
	fig.set_size_inches(14, 10)
	
	# Ajusta el tamaño de la figura y margen inferior
	#plt.figure(figsize=(10, 8))
	#plt.subplots_adjust(bottom=0.2)

	# Agrega título
	#plt.title('Matriz de correlación')

	# Muestra el mapa de calor
	#plt.show()
	plt.savefig(f'{output_folder}/pearson.png')

	plt.close(fig)

def CorrelationSpearman(output_folder,ruta,columnas_analysis):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

	#Cargar el dataframe
	df = pd.read_csv(ruta)

	#Eliminar los valores faltantes
	df=df.dropna()
	#Filtrar las columnas a analizar
	df_correlation=df.loc[:,columnas_analysis]
	# Configura una figura antes del bucle
	fig = plt.figure()

	spearman_corr = df_correlation.corr(method='spearman')

	# Imprime la matriz de correlación
	print(spearman_corr)

	# Opcional: Puedes guardar la matriz de correlación en un archivo CSV
	
	spearman_corr.to_csv(f'{output_folder}/spearman.csv', index=True)

	#filename = f'{output_folder}/MatrixCorrelacion.png'
	    
	# Genera el mapa de calor de la matriz de correlación
	heatmap = sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', square=True)

	# Rotación de los nombres de las columnas
	heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=20, horizontalalignment='right')

	fig = plt.gcf()
	fig.set_size_inches(14, 10)
	
	# Ajusta el tamaño de la figura y margen inferior
	#plt.figure(figsize=(10, 8))
	#plt.subplots_adjust(bottom=0.2)

	# Agrega título
	#plt.title('Matriz de correlación')

	# Muestra el mapa de calor
	#plt.show()
	plt.savefig(f'{output_folder}/spearman.png')

	plt.close(fig)


def CorrelationKendall(output_folder,ruta,columnas_analysis):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

	#Cargar el dataframe
	df = pd.read_csv(ruta)

	#Eliminar los valores faltantes
	df=df.dropna()
	#Filtrar las columnas a analizar
	df_correlation=df.loc[:,columnas_analysis]
	# Configura una figura antes del bucle
	fig = plt.figure()

	kendall_corr = df_correlation.corr(method='kendall')

	# Imprime la matriz de correlación
	print(kendall_corr)

	# Opcional: Puedes guardar la matriz de correlación en un archivo CSV
	
	kendall_corr.to_csv(f'{output_folder}/kendall.csv', index=True)

	#filename = f'{output_folder}/MatrixCorrelacion.png'
	    
	# Genera el mapa de calor de la matriz de correlación
	heatmap = sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', square=True)

	# Rotación de los nombres de las columnas
	heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=20, horizontalalignment='right')

	fig = plt.gcf()
	fig.set_size_inches(14, 10)
	
	# Ajusta el tamaño de la figura y margen inferior
	#plt.figure(figsize=(10, 8))
	#plt.subplots_adjust(bottom=0.2)

	# Agrega título
	#plt.title('Matriz de correlación')

	# Muestra el mapa de calor
	#plt.show()
	plt.savefig(f'{output_folder}/kendall.png')

	plt.close(fig)
def Outliers(output_folder,ruta,columnas_analysis):
	    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(ruta)

    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Iterar sobre las columnas de análisis
    for col in columnas_analysis:
        # Crear una figura y ejes para la gráfica
        fig, ax = plt.subplots(figsize=(8, 6))

        # Generar el boxplot de la columna
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Outliers en la columna '{col}'")

        # Guardar la gráfica en un archivo
        output_file = os.path.join(output_folder, f"{col}_outlier.png")
        plt.savefig(output_file)

        # Cerrar la figura
        plt.close(fig)

#Columnas a analizar
columnas_analysis_OMP_RECURSIVO=['Time','PorCiento_Uso_Memoria','Promedio_Frecuencia_Proc','Promedio_Temp','Max_Temp','Max_Frecuencia','N','Recur','Hilos']

columnas_analysis_OMP_NO_RECURSIVO=['Time','PorCiento_Uso_Memoria','Promedio_Frecuencia_Proc','Promedio_Temp','Max_Temp','Max_Frecuencia','N','Hilos']

columnas_analysis_SECUENCIAL=['Time','PorCiento_Uso_Memoria','Promedio_Frecuencia_Proc','Promedio_Temp','Max_Temp','Max_Frecuencia','N','Recur']

columnas_analysis_CUDA_RECURSIVO=['Time','Promedio_Frecuencia_Proc','Max_Temp','Max_Frecuencia','N','Recur']

columnas_analysis_CUDA_OMP=['Time','Promedio_Frecuencia_Proc','Max_Temp','Max_Frecuencia','N','Recur','Hilos']



# Crea la carpeta para almacenar las gráficas si no existe
output_folderDistribucion_OMP_RECURSIVO = 'GráficasDistribución/DISTRIBUCION_OMP_RECURSIVO'
output_folderDistribucion_OMP_NO_RECURSIVO = 'GráficasDistribución/DISTRIBUCION_OMP_NO_RECURSIVO'
output_folderDistribucion_SECUENCIAL = 'GráficasDistribución/DISTRIBUCION_SECUENCIAL'
output_folderDistribucion_CUDA_OMP = 'GráficasDistribución/DISTRIBUCION_CUDA_OMP_RECURSIVO'
output_folderDistribucion_CUDA_RECURSIVO = 'GráficasDistribución/DISTRIBUCION_CUDA_RECURSIVO'


output_folderCORRELATION_OMP_RECURSIVO = 'CORRELATION/CORRELATION_OMP_RECURSIVO'
output_folderCORRELATION_OMP_NO_RECURSIVO = 'CORRELATION/CORRELATION_OMP_NO_RECURSIVO'
output_folderCORRELATION_SECUENCIAL = 'CORRELATION/CORRELATION_SECUENCIAL'
output_folderCORRELATION_CUDA_OMP = 'CORRELATION/CORRELATION_CUDA_OMP_RECURSIVO'
output_folderCORRELATION_CUDA_RECURSIVO = 'CORRELATION/CORRELATION_CUDA_RECURSIVO'


rutaCUDA_OMP='../../DATOS_CUDA_OPENMP/cuda_omp_recursivo_completo.csv'
rutaCUDA_RECURSIVO='../../DATOS_CUDA_RECURSIVO/cuda_recursivo_completo.csv'
rutaOMP_RECURSIVO='../../DATOS_UNIDOS_OMP_RECURSIVO/UNION_FINAL/UNION_OMP_RECURSIVO.csv'
rutaOMP_NO_RECURSIVO='../../DATOS_UNIDOS_OMP_NO_RECURSIVO/UNION_SERVER_OMP_NO_RECURSIVO.csv'
rutaSECUENCIAL='../../DATOS_UNIDOS_SECUENCIAL/UNION_SERVER_SECUENCIAL.csv'


output_OUTLIERS_SECUENCIAL = 'Outliers/OUTLIERS_SECUENCIAL/'
output_OUTLIERS_CUDA_RECURSIVO = 'Outliers/OUTLIERS_CUDA_RECURSIVO/'
output_OUTLIERS_CUDA_OMP = 'Outliers/OUTLIERS_CUDA_OMP/'
output_OUTLIERS_OMP_NO_RECURSIVO = 'Outliers/OUTLIERS_OMP_NO_RECURSIVO/'
output_OUTLIERS_OMP_RECURSIVO = 'Outliers/OUTLIERS_OMP_RECURSIVO/'



Distribucion(output_folderDistribucion_OMP_RECURSIVO,rutaOMP_RECURSIVO,columnas_analysis_OMP_RECURSIVO)
CorrelationSpearman(output_folderCORRELATION_OMP_RECURSIVO,rutaOMP_RECURSIVO,columnas_analysis_OMP_RECURSIVO)
CorrelationPearson(output_folderCORRELATION_OMP_RECURSIVO,rutaOMP_RECURSIVO,columnas_analysis_OMP_RECURSIVO)
CorrelationKendall(output_folderCORRELATION_OMP_RECURSIVO,rutaOMP_RECURSIVO,columnas_analysis_OMP_RECURSIVO)
Outliers(output_OUTLIERS_OMP_RECURSIVO,rutaOMP_RECURSIVO,columnas_analysis_OMP_RECURSIVO)


Distribucion(output_folderDistribucion_OMP_NO_RECURSIVO,rutaOMP_NO_RECURSIVO,columnas_analysis_OMP_NO_RECURSIVO)
CorrelationSpearman(output_folderCORRELATION_OMP_NO_RECURSIVO,rutaOMP_NO_RECURSIVO,columnas_analysis_OMP_NO_RECURSIVO)
CorrelationPearson(output_folderCORRELATION_OMP_NO_RECURSIVO,rutaOMP_NO_RECURSIVO,columnas_analysis_OMP_NO_RECURSIVO)
CorrelationKendall(output_folderCORRELATION_OMP_NO_RECURSIVO,rutaOMP_NO_RECURSIVO,columnas_analysis_OMP_NO_RECURSIVO)
Outliers(output_OUTLIERS_OMP_NO_RECURSIVO,rutaOMP_NO_RECURSIVO,columnas_analysis_OMP_NO_RECURSIVO)



Distribucion(output_folderDistribucion_SECUENCIAL,rutaSECUENCIAL,columnas_analysis_SECUENCIAL)
CorrelationSpearman(output_folderCORRELATION_SECUENCIAL,rutaSECUENCIAL,columnas_analysis_SECUENCIAL)
CorrelationPearson(output_folderCORRELATION_SECUENCIAL,rutaSECUENCIAL,columnas_analysis_SECUENCIAL)
CorrelationKendall(output_folderCORRELATION_SECUENCIAL,rutaSECUENCIAL,columnas_analysis_SECUENCIAL)
Outliers(output_OUTLIERS_SECUENCIAL,rutaSECUENCIAL,columnas_analysis_SECUENCIAL)



Distribucion(output_folderDistribucion_CUDA_OMP,rutaCUDA_OMP,columnas_analysis_CUDA_OMP)
CorrelationSpearman(output_folderCORRELATION_CUDA_OMP,rutaCUDA_OMP,columnas_analysis_CUDA_OMP)
CorrelationPearson(output_folderCORRELATION_CUDA_OMP,rutaCUDA_OMP,columnas_analysis_CUDA_OMP)
CorrelationKendall(output_folderCORRELATION_CUDA_OMP,rutaCUDA_OMP,columnas_analysis_CUDA_OMP)
Outliers(output_OUTLIERS_CUDA_OMP,rutaCUDA_OMP,columnas_analysis_CUDA_OMP)



Distribucion(output_folderDistribucion_CUDA_RECURSIVO,rutaCUDA_RECURSIVO,columnas_analysis_CUDA_RECURSIVO)
CorrelationSpearman(output_folderCORRELATION_CUDA_RECURSIVO,rutaCUDA_RECURSIVO,columnas_analysis_CUDA_RECURSIVO)
CorrelationPearson(output_folderCORRELATION_CUDA_RECURSIVO,rutaCUDA_RECURSIVO,columnas_analysis_CUDA_RECURSIVO)
CorrelationKendall(output_folderCORRELATION_CUDA_RECURSIVO,rutaCUDA_RECURSIVO,columnas_analysis_CUDA_RECURSIVO)
Outliers(output_OUTLIERS_CUDA_RECURSIVO,rutaCUDA_RECURSIVO,columnas_analysis_CUDA_RECURSIVO)





#-------------------------------------------------------------------Análisis de Correlación------------------------------------------------------------------------

