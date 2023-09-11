#Empezar por el secuencial
valor_especifico = 'CUDA'

# Filtrar los datos
#datos_filtrados = df[df['MODEL'] == valor_especifico]
columnas_deseadasForest = ['Time','N','Recur','Max_Consumo_Memoria','Promedio_PorCiento_Uso_Proc','Promedio_Frecuencia_Proc']
columnas_deseadas = ['Time','N','Recur','Hilos','Max_Consumo_Memoria','PorCiento_Uso_Memoria','Promedio_PorCiento_Uso_Proc','Max_PorCiento_Uso_Proc','Promedio_Frecuencia_Proc','Max_Frecuencia']
# Seleccionar las columnas deseadas




datos_seleccionados = df.loc[:, columnas_deseadas]
datos_seleccionadosForest = df.loc[:, columnas_deseadasForest]

"""# Crear una instancia del escalador StandardScaler
   scalers = {}
   # Iterar sobre cada columna y aplicar el scaler correspondiente
   for col in columnas_deseadas:
       # Crear un scaler para la columna actual
       scaler = StandardScaler()
       # Obtener los valores de la columna y reshape para que sea un array 2D
       columna = datos_seleccionados[col].values.reshape(-1, 1)
       # Normalizar la columna actual y guardar el scaler en el diccionario
       columna_normalizada = scaler.fit_transform(columna)
       scalers[col] = scaler
       # Agregar la columna normalizada al DataFrame df_normalizado
       datos_seleccionados[col] = columna_normalizada.flatten()
   """   
#Normalizar utilizando z_score
#df_normalizado = pd.DataFrame(scaler.fit_transform(datos_seleccionados), columns=datos_seleccionados.columns)
#Dividir los datos normalizados en prueba y entrenamiento


X= datos_seleccionados.loc[:, columnas_deseadas[1:len(columnas_deseadas)]]
Y=datos_seleccionados.loc[:,columnas_deseadas[0]]
print(len(columnas_deseadas))
Xf= datos_seleccionadosForest.loc[:, columnas_deseadasForest[1:len(columnas_deseadasForest)]]
Yf=datos_seleccionadosForest.loc[:,columnas_deseadasForest[0]]


X_trainF, X_testF, y_trainF, y_testF = train_test_split(Xf, Yf, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=42)
#Probar con LSTM
# Dividir los datos en conjuntos de entrenamiento y prueba
# Definir el número de pasos de tiempo para la secuencia de entrada
time_steps = 10
# Preparar los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_trainLstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_testLstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_trainLstm = np.reshape(y_train, (-1, 1))
y_testLstm = np.reshape(y_test, (-1, 1))
#print(X_trainLstm)
optimizer=Adagrad(learning_rate=0.01)#regular mse=0.09 r2=0.70
optimizer2=Adadelta(learning_rate=0.01)#malo
optimizer3=RMSprop(learning_rate=0.01)#buenisimo mse=0.01 r2=0.96
optimizer4=SGD(learning_rate=0.01,momentum=0.99)#bueno mse=0.04 r2=0.85
#print(X_trainLstm)
#inpua=input_shape=(1, X.shape[1])
#print(X_trainLstm[0][0])
print(X_train.head())


#Regresiones y evaluaciones

#SecuencialPuro
modelSecuencial = Sequential()
modelSecuencial.add(Dense(9,input_dim=9,kernel_initializer='normal'))
"""modelSecuencial.add(Dense(100,kernel_initializer='normal'))
modelSecuencial.add(Dense(60,kernel_initializer='normal'))
modelSecuencial.add(Dense(100,kernel_initializer='normal'))
"""
modelSecuencial.add(Dense(60,kernel_initializer='normal'))
modelSecuencial.add(Dense(1,kernel_initializer='normal'))
modelSecuencial.compile(loss='mean_squared_error', optimizer=optimizer3)
modelSecuencial.fit(X_train, y_train, epochs=60, batch_size=32)

# Evaluar el modelo con los datos de prueba
result = modelSecuencial.evaluate(X_test, y_test)  # Ajusta los datos de prueba adecuados
print("Modelo Secuencial Puro"+str(result))
# Crear un modelo secuencial para LSTM
model = Sequential()
# Agregar una capa LSTM
model.add(LSTM(units=80, input_shape=(1, X.shape[1])))
# Agregar una capa de salida
model.add(Dense(units=1))
# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer=optimizer3)

# Entrenar el modelo
model.fit(X_trainLstm, y_train, epochs=60, batch_size=32)

# Evaluar el modelo en los datos de prueba
mse = model.evaluate(X_testLstm, y_test)
print("Error cuadrático medio (MSE):", mse)
#Probar con Random Forest
model_regression.fit(X_trainF, y_trainF)
score=model_regression.score(X_testF,y_testF)
predictionsRandomForest=model_regression.predict(X_testF)
mae = mean_absolute_error(y_testF, predictionsRandomForest)
mape = np.mean(np.abs((y_testF - predictionsRandomForest) / y_testF)) * 100

print("Random Forest: " + str(score))

print("Random Forest: " + str(mae))

print("Random Forest: " + str(mape))

predictionsLstm = model.predict(X_testLstm)
#y_testLstm=np.reshape(-1, 1)
#print(y_test)
desnormalized_predictionsLstm=scalers['Time'].inverse_transform(predictionsLstm)
reshape_forest_prediction = np.reshape(predictionsRandomForest, (-1, 1))
desnormalized_real_values=scalers['Time'].inverse_transform(y_testLstm)
desnormalized_predictionsRandomForest=scalers['Time'].inverse_transform(reshape_forest_prediction)
print("LSTM Predicciones")
print(desnormalized_predictionsLstm)
print("Valores Reales")
print(y_testF)
print("Random Forest Predicciones")
print(predictionsRandomForest)
r2 = r2_score(y_testLstm, predictionsLstm)
print("Coeficiente de determinación (R cuadrado) para LSTM:", r2)



