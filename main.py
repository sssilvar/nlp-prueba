#!/bin/env python3
import sys
from os.path import realpath

import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from pd2img.converter import dataframe_to_image

sys.path.append(realpath(__file__))
sns.set()


def extract_data_vectors(filename, training_sheets, testing_sheets):
    """
    Permite extraer dos conjuntos a partir de el archivo de Excel que contiene las respuestas y las clases.
    :param filename: Archivo Excel que contenga los datos.
    :param training_sheets: Lista de hojas que contienen los datos de entrenamiento
    :param testing_sheets: Lista de hojas que contienen los datos de validación
    :return: (tuple) X_train, y_train, X_test.
             X contiene el texto
             y contiene las etiquetas (y_test no es proveído)
    """
    # Crear un DataFrame de entrenamiento y prueba vacíos
    df_train = df_test = pd.DataFrame()
    xls = pd.ExcelFile(filename)  # Objeto Excel

    # Leer datos de entrenamiento
    for sheet in training_sheets:
        df = pd.read_excel(xls, sheet)
        df_train = df_train.append(df, ignore_index=True, sort=False)

    # Leer datos de prueba
    for sheet in testing_sheets:
        df = pd.read_excel(xls, sheet)
        df_test = df_train.append(df, ignore_index=True, sort=False)

    X_train, y_train = df_train['P6390'], df_train['RAMA2D_R4']
    X_test = df_test['P6390']
    return X_train, y_train, X_test


def create_pipeline():
    """
    Derinir un pipeline para preprocesamiento, extracción de características y clasificación del texto.
    :return: Sklearn Pipeline
    """
    # Crear un vectorizador para extraer características del texto de entrada (respuestas).
    vectorizer = TfidfVectorizer(analyzer='word',
                                 token_pattern='\w{1,}',
                                 max_features=5000)

    # Crear un clasificador que se encarge de aprender y asignar las características del texto.
    # (Random Forest)
    clf = RandomForestClassifier(n_estimators=10)

    # Crear y retornar el pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', clf)
    ])
    return pipeline


if __name__ == '__main__':
    # Leer el archivo Excel que contiene las respuestas
    # y crear dos dataframes (entrenamiento/validación)
    X, y, X_test = extract_data_vectors(filename='Datos Ejercicio CIIURev4.xlsx',
                                        training_sheets=['mes1', 'mes2', 'mes3'],
                                        testing_sheets=['validacion'])
    df_train = pd.concat([X, y], axis='columns')
    print(df_train.head())

    # Instanciar y entrenar pipeline
    pipeline = create_pipeline()
    pipeline.fit(X, y)

    # Validar modelo utilizando datos de prueba
    # (Asignación de etiquetas): y_pred
    y_pred = pipeline.predict(X_test)
    y_pred = pd.Series(y_pred, name='RAMA2D_R4')

    df_pred = pd.concat([X_test, y_pred], axis='columns')

    # Analizar resultados encontrados por categoría.
    categories = y.value_counts().index
    sep = '='
    for category in categories:
        df_test_head = df_pred.query(f'RAMA2D_R4 == {category}').head()
        df_train_head = df_train.query(f'RAMA2D_R4 == {category}').head()

        print(f'{sep * 10} PPREDICTED {sep * 10}')
        print(df_test_head)
        print(f'{sep * 10} GROUNDTRUTH {sep * 10}')
        print(df_train_head)
        print('\n\n')

        try:
            dataframe_to_image(data=df_test_head,
                               outputfile=f'/tmp/{category}_test.png')
            dataframe_to_image(data=df_test_head,
                               outputfile=f'/tmp/{category}_train.png')
        except:
            pass
