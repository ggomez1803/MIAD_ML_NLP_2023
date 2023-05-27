#!/usr/bin/python

import pandas as pd
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import ast
import joblib
import sys
import os

# Funciones
# Definición de la función que tenga como parámetro texto y devuelva una lista de lemas
def split_into_lemmas(text):
    text = text.lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word) for word in words]

# Columnas de respuesta
cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

# Definición de variables predictoras (X)
dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)

# Convertir columna géneros de string a lista
dataTraining['genres'] = dataTraining['genres'].apply(lambda x: ast.literal_eval(x))

# Crear MultilabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit(dataTraining['genres'])

# Transformar columna géneros en array binario
y_genres = mlb.transform(dataTraining['genres'])

# Crear objeto CountVectorizer y entrenar
vectorizer = CountVectorizer(stop_words='english', max_features=200000, analyzer=split_into_lemmas)
X_dtm = vectorizer.fit_transform(dataTraining['plot'])

# Función de predicción
def predict_genre(title, plot):
    # Cargarmodelo
    model = joblib.load(os.path.dirname(__file__) + '/ovr_clf.pkl') 
    # Crear el dataframe para introducir al modelo de predicción
    x_test = pd.DataFrame(columns=['title', 'plot'])
    x_test.loc[0,'plot'] = plot
    x_test.loc[0,'title'] = title
    # Transformar plot en array binario
    x_dtm = vectorizer.transform(x_test['plot'])
    # Predecir
    y_pred_prob = model.predict_proba(x_dtm)
    y_pred = model.predict(x_dtm)
    # Transformar predicción en géneros
    y_pred_genres = mlb.inverse_transform(y_pred)
    return y_pred_genres

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Por favor ingresar todos los argumentos de la película')
    else:
        # Leer argumentos
        title = sys.argv[1]
        plot = sys.argv[2]
        # Predecir
        y_pred_genres = predict_genre(title, plot)
        # Imprimir géneros
        print('Posibles géneros:', y_pred_genres)