#!/bin/env python3
import os
import sys
from os.path import realpath, join, isfile, dirname

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from pd2img.converter import dataframe_to_image

# Carpeta raíz
root = dirname(realpath(__file__))
sys.path.append(root)

# Descargar diccionarios
nltk.download('stopwords')
nltk.download('punkt')


def remove_punctuation(text):
    """
    Función para remover los signos de puntiación y carácteres especiales.
    :param text: (str) contiene el texto a preprocesar.
    :return: (str) texto sin signos.
    """
    no_punct = ''.join([c for c in text if c not in punctuation])
    return no_punct


def tokenize(text):
    no_punct = remove_punctuation(text)
    tokens = nltk.word_tokenize(no_punct, language='spanish')
    stems = []
    for item in tokens:
        stems.append(SnowballStemmer("spanish").stem(item))
    return stems


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
    df_train = pd.DataFrame()
    xls = pd.ExcelFile(filename)  # Objeto Excel

    # Leer datos de entrenamiento
    for sheet in training_sheets:
        df = pd.read_excel(xls, sheet)
        df_train = df_train.append(df, ignore_index=True, sort=False)

    # Leer datos de prueba
    df_test = pd.DataFrame()
    for sheet in testing_sheets:
        df = pd.read_excel(xls, sheet)
        df_test = df_test.append(df, ignore_index=True, sort=False)

    X_train, y_train = df_train['P6390'], df_train['RAMA2D_R4']
    X_test = df_test['P6390']
    return X_train, y_train, X_test


def create_pipeline():
    """
    Derinir un pipeline para preprocesamiento, extracción de características y clasificación del texto.
    :return: Sklearn Pipeline
    """
    # Crear un vectorizador para extraer características del texto de entrada (respuestas).
    # vectorizer = TfidfVectorizer(analyzer='word',
    #                              token_pattern='\w{1,}',
    #                              max_features=5000)
    vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=5000)

    # Crear un clasificador que se encarge de aprender y asignar las características del texto.
    # (Random Forest)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)

    # Crear y retornar el pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', clf)
    ])
    return pipeline


def extract_sample(dataframe):
    """
    Extrae una muestra aleatoria si la cantidad de observaciones por cada clase es mayor a 5.
    :param dataframe: Pandas DataFrame
    :return: (pandas.DataFrame) muestra aleatoria de 5 elementos
    """
    if len(dataframe) > 5:
        return dataframe.sample(n=5, random_state=21)
    else:
        return dataframe.head()


if __name__ == '__main__':
    # Definir ruta de archivo excel al mismo nivel del script.
    excel_file = join(root, 'Datos Ejercicio CIIURev4.xlsx')
    print(f'Data file: {excel_file}')
    assert isfile(excel_file), 'Excel data file was not found.'  # Verificar existencia del archivo Excel

    # Carpeta de resultados
    out_folder = join(root, 'results')
    os.makedirs(out_folder, exist_ok=True)

    # Crear imágenes de los DataFrames (para poner en presentación)
    create_images = False

    # Leer el archivo Excel que contiene las respuestas
    # y crear dos dataframes (entrenamiento/validación)
    X, y, X_test = extract_data_vectors(filename=excel_file,
                                        training_sheets=['mes1', 'mes2', 'mes3'],
                                        testing_sheets=['validacion'])
    df_train = pd.concat([X, y], axis='columns')
    print(df_train.head())
    print(f'Training dataset samples: {len(X)}')
    print(f'Testing dataset samples: {len(X_test)}')
    # exit()

    # Instanciar y entrenar pipeline
    pipeline = create_pipeline()
    print('Training pipeline...')
    pipeline.fit(X, y)

    # Validar modelo utilizando datos de prueba
    # (Asignación de etiquetas): y_pred
    print('Testing pipeline...')
    y_pred = pipeline.predict(X_test)
    y_pred = pd.Series(y_pred, name='RAMA2D_R4')

    # Crear Dataframe con categorías predecidas
    df_pred = pd.concat([X_test, y_pred], axis='columns')

    # Guardar Excel con predicciones
    df_pred.to_excel(join(out_folder, 'predicciones.xlsx'))

    # Graficar distribución
    plt.figure(figsize=(20, 8))
    ordered_labels = df_pred['RAMA2D_R4'].value_counts().index
    sns.countplot(x='RAMA2D_R4', data=df_pred, order=ordered_labels)
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.savefig(join(out_folder, 'result_distribition.png'), bbox_inches='tight')

    # Analizar resultados encontrados por categoría.
    categories = y.value_counts().index
    sep = '='
    for category in categories:
        # Filtrar por categoría
        df_test_query = df_pred.query(f'RAMA2D_R4 == {category}')
        df_train_query = df_train.query(f'RAMA2D_R4 == {category}')

        # Extraer muestra aleatoria si hay más de 5
        # observaciones en la clase
        df_test_head = extract_sample(df_test_query)
        df_train_head = extract_sample(df_train_query)

        # Imprimir resultados
        print(f'{sep * 10} PPREDICTED {sep * 10}')
        print(df_test_head)
        print(f'{sep * 10} GROUNDTRUTH {sep * 10}')
        print(df_train_head)
        print('\n\n')

        # Guardar dataframes como imágenes
        # (para reporte y presentación)
        if create_images:
            try:
                dataframe_to_image(data=df_test_head,
                                   outputfile=join(out_folder, f'{category}_test.png'))
                dataframe_to_image(data=df_train_head,
                                   outputfile=join(out_folder, f'{category}_train.png'))
            except:
                pass
