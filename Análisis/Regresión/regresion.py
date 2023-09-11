import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import r2_score
from keras.optimizers import Adagrad 
from keras.optimizers import Adadelta 
from keras.optimizers import RMSprop
from keras.optimizers import SGD 
from sklearn.metrics import mean_absolute_error

import numpy as np

import os


def RegressorRandomForest(output_folder,file_path,columnas_deseadas,name):
	pass
	#Importar Random Forest
	random_forest=RandomForestRegressor(random_state=42)
	#Cargar el dataframe
	df=pd.read_csv(file_path)
	df.fillna(0, inplace=True)
	#Buscar las columnas a analizar
	datos_seleccionadosForest = df.loc[:, columnas_deseadasForest]
	#División variable objetivo y variable predictora
	Xf= datos_seleccionadosForest.loc[:, columnas_deseadasForest[1:len(columnas_deseadasForest)]]
	Yf=datos_seleccionadosForest.loc[:,columnas_deseadasForest[0]]
	#División datos de entrenamiento y datos de prueba
	X_trainF, X_testF, y_trainF, y_testF = train_test_split(Xf, Yf, test_size=0.2, random_state=42)

	#Entrenamiento con los datos de entrenamiento
	random_forest.fit(X_trainF, y_trainF)

	#Evaluaciones
	score=random_forest.score(X_testF,y_testF)
	predictionsRandomForest=random_forest.predict(X_testF)
	mae = mean_absolute_error(y_testF, predictionsRandomForest)
	mape = np.mean(np.abs((y_testF - predictionsRandomForest) / y_testF)) * 100

	print("RandomForest_R2",score)

	"""# Nombres de los valores
				nombres_valores = ['Score', 'MAE', 'MAPE']
			
				# Rango de los valores en el eje x
				x = range(len(nombres_valores))
				
				# Valores a graficar en el eje y
				y = [score, mae, mape]
			
				# Graficar los valores
				plt.bar(x, y)
				plt.xticks(x, nombres_valores)
				plt.ylabel('Valor')
				plt.title('Evaluaciones del modelo Random Forest')
			
				# Mostrar la gráfica
				plt.show()
			"""

	#Empezar por el secuencial
	valor_especifico = 'CUDA'

	# Filtrar los datos
	#datos_filtrados = df[df['MODEL'] == valor_especifico]

#Creación de Carpeta de Resultados e importación de los Datos con Panda


file_path_CUDA_OMP_RECURSIVO = '../Preprocesamiento/DATOS_PREPROCESADOS/DATOS_CUDA_OMP_RECURSIVO/preprocessed_data_CUDA_OMP_RECURSIVO.csv'
file_path_CUDA_RECURSIVO = '../Preprocesamiento/DATOS_PREPROCESADOS/DATOS_CUDA_RECURSIVO/preprocessed_data_CUDA_RECURSIVO.csv'

file_path_OMP_NO_RECURSIVO = '../Preprocesamiento/DATOS_PREPROCESADOS/DATOS_OMP_NO_RECURSIVO/preprocessed_data_OMP_NO_RECURSIVO.csv'
file_path_OMP_RECURSIVO = '../Preprocesamiento/DATOS_PREPROCESADOS/DATOS_OMP_RECURSIVO/preprocessed_data_OMP_RECURSIVO.csv'
file_path_SECUENCIAL = '../Preprocesamiento/DATOS_PREPROCESADOS/DATOS_SECUENCIAL/preprocessed_data_SECUENCIAL.csv'

#Guardar los datos en dataframe->df
df_CUDA_OMP_RECURSIVO= pd.read_csv(file_path_CUDA_OMP_RECURSIVO)
df_CUDA_RECURSIVO= pd.read_csv(file_path_CUDA_RECURSIVO)
df_OMP_RECURSIVO= pd.read_csv(file_path_OMP_RECURSIVO)
df_OMP_NO_RECURSIVO= pd.read_csv(file_path_OMP_NO_RECURSIVO)
df_SECUENCIAL= pd.read_csv(file_path_SECUENCIAL)


RESULT_FOLDER='Resultados_Modelo/'

columnas_deseadasForest = ['Time','N','Recur','Promedio_PorCiento_Uso_Proc','Max_Temp']

RegressorRandomForest(RESULT_FOLDER,file_path_SECUENCIAL,columnas_deseadasForest,'OMP_RECURSIVO')



