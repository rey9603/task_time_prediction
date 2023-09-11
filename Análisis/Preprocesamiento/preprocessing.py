import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def one_hot_encoding(data, columns):
    # Realizar la codificación one-hot de las columnas seleccionadas
    encoder = OneHotEncoder()
    encoded_cols = pd.DataFrame(encoder.fit_transform(data[columns]).toarray(), columns=encoder.get_feature_names_out(columns))
    data = pd.concat([data, encoded_cols], axis=1)
    data = data.drop(columns, axis=1)
    return data

def standard_scaling(data, columns):
    # Realizar la normalización usando StandardScaler de las columnas seleccionadas
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def preprocess_data(output_folder, file_path, categorical_columns, numerical_columns,name):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(file_path)

    # Aplicar codificación one-hot a las columnas categóricas
    if categorical_columns:
        data = one_hot_encoding(data, categorical_columns)

    # Aplicar normalización a las columnas numéricas
    if numerical_columns:
        data = standard_scaling(data, numerical_columns)

    # Guardar los datos preprocesados en un nuevo archivo CSV
    output_path = f'{output_folder}/preprocessed_data_{name}.csv'
    data.to_csv(output_path, index=False)
    print(f"Los datos preprocesados se han guardado en: {output_path}")

def preprocess_data_CUDA(output_folder, file_path, numerical_columns,name):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(file_path)

    """# Aplicar codificación one-hot a las columnas categóricas
                if categorical_columns:
                    data = one_hot_encoding(data, categorical_columns)
            """
    # Aplicar normalización a las columnas numéricas
    if numerical_columns:
        data = standard_scaling(data, numerical_columns)

    # Guardar los datos preprocesados en un nuevo archivo CSV
    output_path = f'{output_folder}/preprocessed_data_{name}.csv'
    data.to_csv(output_path, index=False)
    print(f"Los datos preprocesados se han guardado en: {output_path}")


output_folder_preprocessing_CUDA_OMP_RECURSIVO = 'DATOS_PREPROCESADOS/DATOS_CUDA_OMP_RECURSIVO'
file_path_CUDA_OMP_RECURSIVO = '../../DATOS_CUDA_OPENMP/cuda_omp_recursivo_completo.csv'
numerical_columns_CUDA_OMP_RECURSIVO  = ['Time','Promedio_Frecuencia_Proc','Max_Temp','Max_Frecuencia','N','Recur','Hilos']

output_folder_preprocessing_CUDA_RECURSIVO = 'DATOS_PREPROCESADOS/DATOS_CUDA_RECURSIVO'
file_path_CUDA_RECURSIVO = '../../DATOS_CUDA_RECURSIVO/cuda_recursivo_completo.csv'
numerical_columns_CUDA_RECURSIVO = ['Time','Promedio_Frecuencia_Proc','Max_Temp','Max_Frecuencia','N','Recur']

output_folder_preprocessing_OMP_RECURSIVO = 'DATOS_PREPROCESADOS/DATOS_OMP_RECURSIVO'
file_path_OMP_RECURSIVO = '../../DATOS_UNIDOS_OMP_RECURSIVO/UNION_FINAL/UNION_OMP_RECURSIVO.csv'
categorical_columns_OMP_RECURSIVO = ['Tipo_Procesador','Tipo_Memoria','Tipo_Disco' , 'ESTRATEGIA']
numerical_columns_OMP_RECURSIVO = ['Time','PorCiento_Uso_Memoria','Promedio_Frecuencia_Proc','Promedio_Temp','Max_Temp','Max_Frecuencia','N','Recur','Hilos']

output_folder_preprocessing_OMP_NO_RECURSIVO = 'DATOS_PREPROCESADOS/DATOS_OMP_NO_RECURSIVO'
file_path_OMP_NO_RECURSIVO = '../../DATOS_UNIDOS_OMP_NO_RECURSIVO/UNION_SERVER_OMP_NO_RECURSIVO.csv'
categorical_columns_OMP_NO_RECURSIVO = ['Tipo_Procesador','Tipo_Memoria','Tipo_Disco' , 'ESTRATEGIA']
numerical_columns_OMP_NO_RECURSIVO = ['Time','PorCiento_Uso_Memoria','Promedio_Frecuencia_Proc','Promedio_Temp','Max_Temp','Max_Frecuencia','N','Hilos']

output_folder_preprocessing_SECUENCIAL = 'DATOS_PREPROCESADOS/DATOS_SECUENCIAL'
file_path_SECUENCIAL = '../../DATOS_UNIDOS_SECUENCIAL/UNION_SERVER_SECUENCIAL.csv'
categorical_columns_SECUENCIAL = ['Tipo_Procesador','Tipo_Memoria','Tipo_Disco' , 'ESTRATEGIA']
numerical_columns_SECUENCIAL = ['Time','PorCiento_Uso_Memoria','Promedio_Frecuencia_Proc','Promedio_Temp','Max_Temp','Max_Frecuencia','N','Recur']


preprocess_data(output_folder_preprocessing_OMP_RECURSIVO, file_path_OMP_RECURSIVO, categorical_columns_OMP_RECURSIVO, numerical_columns_OMP_RECURSIVO,'OMP_RECURSIVO')
preprocess_data(output_folder_preprocessing_OMP_NO_RECURSIVO, file_path_OMP_NO_RECURSIVO, categorical_columns_OMP_NO_RECURSIVO, numerical_columns_OMP_NO_RECURSIVO,'OMP_NO_RECURSIVO')
preprocess_data(output_folder_preprocessing_SECUENCIAL, file_path_SECUENCIAL, categorical_columns_SECUENCIAL, numerical_columns_SECUENCIAL,'SECUENCIAL')
preprocess_data_CUDA(output_folder_preprocessing_CUDA_OMP_RECURSIVO, file_path_CUDA_OMP_RECURSIVO,numerical_columns_CUDA_OMP_RECURSIVO,'CUDA_OMP_RECURSIVO')
preprocess_data_CUDA(output_folder_preprocessing_CUDA_RECURSIVO, file_path_CUDA_RECURSIVO,numerical_columns_CUDA_RECURSIVO,'CUDA_RECURSIVO')







