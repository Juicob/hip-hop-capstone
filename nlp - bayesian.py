# %%
import os
import sys
from time import time
# import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import string
import re

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import word2vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import recall_score, confusion_matrix, classification_report, accuracy_score, plot_confusion_matrix

from sklearn import metrics

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.preprocessing import text, sequence

from gensim.models import Phrases
from gensim.models.phrases import Phraser

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# %%
def plot_results(results):
  """Function to convert a models results into a dataframe and plot them to show the both the accuracy and validation accuracy, as well as the loss and validation loss over epochs.

  Args:
      results_dataframe (dataframe): 
  """

  results_dataframe = pd.DataFrame(results)

  fig = px.line(results_dataframe, x=results_dataframe.index, y=["accuracy","val_accuracy"])
  fig.update_layout(title='Accuracy and Validation Accuracy over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Percentage',
                )
  fig.update_traces(mode='lines+markers')
  fig.show()

  fig = px.line(results_dataframe, x=results_dataframe.index, y=['loss','val_loss'])
  fig.update_layout(title='Loss and Validation Loss over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='idek what this unit is - change me'
                )
  fig.update_traces(mode='lines+markers')
  fig.show()

def plotImages(images_arr, labels_arr):
    labels_arr = ['Normal: 0' if label == 0 else 'Pneumonia: 1' for label in labels_arr]
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, label, ax in zip(images_arr, 
                              labels_arr, 
                              axes):
        ax.imshow(img)
        ax.set_title(label, size=18)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_results(model): 
    labels = ['East','South', 'Mid-West', 'West']
    predictions = model.predict(X_test).argmax(axis=1)
    cm = metrics.confusion_matrix(y_test.argmax(axis=1), 
                                    predictions,
                                    normalize="pred")

    ax = sns.heatmap(cm, cmap='Blues',annot=True,square=True)
    ax.set(xlabel='Predicted Class',ylabel='True Class')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    print(metrics.classification_report(y_test.argmax(axis=1), predictions))

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def plot_conf_matrix(y_true, y_pred):
    
    """
    Plots a confusion matrix and displays classification report.
    """
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='0.2g', annot_kws={"size": 14},
                xticklabels=nb.classes_, yticklabels=nb.classes_, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def evaluate_model(model, X_train, X_test):
    y_preds_train = model.predict(X_train.todense())
    y_preds_test = model.predict(X_test.todense())

    # print('Training Accuracy:', accuracy(y_train, y_preds_train, average='weighted'))
    print('Training Accuracy:', accuracy_score(y_train, y_preds_train))
    print('Testing Accuracy:', accuracy_score(y_test, y_preds_test))
    print('Training Recall:', recall_score(y_train, y_preds_train, average='weighted'))
    print('Testing Recall:', recall_score(y_test, y_preds_test, average='weighted'))
    print('\n---------------\n')
    print('Train Confusion Matrix\n')
    plot_conf_matrix(y_train, y_preds_train)
    print('Test Confusion Matrix\n')
    plot_conf_matrix(y_test, y_preds_test)
    print('\n----------------\n')
    print(classification_report(y_test, y_preds_test))

# %%
df = pd.read_csv('../capstone-data/lyrics/lyrics_main.csv',encoding='unicode_escape')
print('-- Dataframe shape: ',df.shape)
print('-- Number of duplicate rows: ',df.duplicated().sum())
print('\n-- Checking for unexpected NUll values-\n',df.isna().sum())
df.head()

# %% [markdown]
# number of artists
# See how many tracks per artist
# albums per artist
# number of songs with features
# most popular featured artists
# most popular producers
# most popular songs - need to convert track views
# %%
artists_to_clean = [
'2 Chainz 8,145',
'Chance the Rapper 47,380',
'Childish Gambino 4,621',
'Common 10,680',
'E-40 2,486',
'Future 4,246',
'Ice Cube 2,636',
'J. Cole 505',
'JAY-Z 175',
'Jeezy 4,855',
'Lil Wayne 4,188',
'Lupe Fiasco 1,619',
'Mac Miller 37,632',
'Nas 37,937',
'Nipsey Hussle 3,475',
'Rick Ross 4,992',
'Royce da 5\'9\" 15,776',
'Snoop Dogg 4,935',
'T.I. 7,556',
'The Game 1,170',
'Too $hort 3,022',
'Travis Scott 3,948'
]


df.artist = df.artist.apply(lambda x: x.replace(x, str(x.split(' ')[:-1])).replace("['",'').replace("']",'').replace("', '", " ").replace('\\','') if x in artists_to_clean else x)
# %%
df_trimmed = df[['region', 'lyrics']]
# %%
df, df_test = train_test_split(df_trimmed, test_size=.2, shuffle=True, random_state=42)
# %%
def clean_lyrics(lyrics):
    
    ## Convert words to lower case and split them
    lyrics = lyrics.lower().split()
    
    ## Remove stop words
    stopwords_list = stopwords.words("english")
    stopwords_list += string.punctuation
    lyrics = [w for w in lyrics if not w in stopwords_list]
    # stopword_list = stopwords.words('english')
    # stopword_list += string.punctuation
    
    lyrics = " ".join(lyrics)
    ## Clean the lyrics
    lyrics = re.sub(r"\[[^\]]*\]", " ", lyrics)
    lyrics = re.sub(r"\n", " ", lyrics)
    lyrics = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", lyrics)
    lyrics = re.sub(r"  "," ", lyrics)
    return lyrics

# %%

# %%
# reg_brackets = '\[[^\]]*\]'
# reg_newline = '\n'
# df.lyrics[1][400:500]
# df.lyrics.to_list()
# re.sub(reg_brackets, ' ', ','.join(corpus))
# re.sub(reg_newline, ' ', corpus)
df.lyrics[7]
# %%
# df.lyrics = df.lyrics.apply(lambda x: re.sub('\[[^\]]*\]', ' ', x))
df.lyrics = df.lyrics.apply(lambda x: x.replace('\n', ' \n '))
df.lyrics = df.lyrics.apply(lambda x: x.replace('\n', 'newline'))
df.lyrics = df.lyrics.apply(lambda x: clean_lyrics(x))
df['lemmatized_lyrics'] = df.lyrics.apply(lambda x: WordNetLemmatizer().lemmatize(x))
# %%
display(df.lemmatized_lyrics[7])
display(df.lyrics[7])

# %%
# stopwords = []
# documents = list(df[df['school'] == 'german_idealism']['gensim_tokenized'])
# sentences = [sentence for sentence in documents]
# cleaned = []
# for sentence in sentences:
#   cleaned_sentence = [word.lower() for word in sentence]
#   cleaned_sentence = [word for word in sentence if word not in stopwords]
#   cleaned.append(cleaned_sentence)
df.lyrics = df.lyrics.apply(lambda x: x.split('newline'))
df.lemmatized_lyrics = df.lemmatized_lyrics.apply(lambda x: x.split('newline'))
bigram = Phrases(df.lyrics, min_count=1, threshold=10, delimiter=b'__')

bigram_phraser = Phraser(bigram)
# tokens_list = []
# for sent in df.lyrics:
#     tokens_ = bigram_phraser[sent]
df['phrases'] = df.lyrics.apply(lambda x:bigram_phraser[x])
#     tokens_list.append(tokens_)

df.lyrics = df.lyrics.apply(lambda x:' '.join(x))
df.lemmatized_lyrics = df.lemmatized_lyrics.apply(lambda x:' '.join(x))
df.phrases = df.phrases.apply(lambda x:' '.join(x))
# %%
display(df.lemmatized_lyrics[7])
display(df.lyrics[7])
# %%
df.shape
# %%
# ! might need to move this for after the vec table to feed into the rand forest model
# %%
df.phrases[1]
# %%
df_grouped = df.groupby(by='region').agg(lambda x:' '.join(x))
# %%
len(df_grouped.lyrics[0])
# %%
df_grouped.lyrics[2]
# %%
# * Play with ngrams for clouds and charts
vectorizer = TfidfVectorizer(analyzer='word')#, ngram_range=(1,2))#decode_error='ignore')
vec_table = vectorizer.fit_transform(df_grouped.lyrics)
vec_table = pd.DataFrame(vec_table.toarray(), columns=vectorizer.get_feature_names())
vec_table.index = df_grouped.index
vec_table
# %%
vec_table
# %%
X_train = vectorizer.fit_transform(df.lyrics)
X_test = vectorizer.transform(df_test.lyrics)
X_train_lemm = vectorizer.fit_transform(df.lemmatized_lyrics)
X_test_lemm = vectorizer.transform(df_test.lyrics)
y_train = df.region
y_test = df_test.region
# %%
nb = MultinomialNB()
nb.fit(X_train.todense(), y_train)
nb_lemm = MultinomialNB()
nb_lemm.fit(X_train_lemm.todense(), y_train)
# %%
rf = RandomForestClassifier()
rf.fit(X_train.todense(), y_train)
rf_lemm = RandomForestClassifier()
rf_lemm.fit(X_train.todense(), y_train)
#
# ! clear out ram before running nb and rb
# %%
rf_param_grid={'max_depth': [1,2,3, None],
            'max_leaf_nodes': [2,3,5,None],
            # 'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [1,2,5],
            'n_estimators': [10,100,300,500],
       #      Unsure if verbose and random_state are needed here but tossed them in for good measure. I wasn't able to get any consistent progress information during training unfortunately so I just left it here
            # 'verbose': [1],
            'random_state':[42]
}
# %%
# %%time
# n_iter_search = 100
# random_search = RandomizedSearchCV(rf, param_distributions=rf_param_grid,
#                                    n_iter=n_iter_search, cv=5, n_jobs=4)

# start = time()
# random_search.fit(X_train, y_train)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
# display(report(random_search.cv_results_))
# display(random_search.best_estimator_)
# %%
# random_best = random_search.best_estimator_
random_best = RandomForestClassifier(min_samples_leaf=2, n_estimators=500, random_state=42)
random_best_lemm = RandomForestClassifier(min_samples_leaf=2, n_estimators=500, random_state=42)
# %%
random_best.fit(X_train.todense(), y_train)
random_best_lemm.fit(X_train_lemm.todense(), y_train)
# %%
 # %%
#  Using the best estimator to refit and and ouput the results
# %%
# display(plot_confusion_matrix(random_search, X_train, y_train, normalize='true', cmap='bone'))
# display(plot_confusion_matrix(random_search, X_test, y_test, normalize='true', cmap='bone'))
print('random_best')
evaluate_model(random_best, X_train, X_test)
print('random_best_lemm')
evaluate_model(random_best_lemm, X_train_lemm, X_test)
# display(plot_confusion_matrix(rf, X_test, y_test, normalize='true', cmap='bone'))
print('baseline random forest')
evaluate_model(rf, X_train, X_test)
print('random forest_lemm')
evaluate_model(rf_lemm, X_train_lemm, X_test)
# %%
# Initializing gridsearch and fitting, and outputting the results and grabbing the best estimator
gridsearch = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rf_param_grid, 
                          scoring='precision', cv=5, verbose=1,n_jobs=-1)
gridsearch.fit(X_train, y_train)
# %%
display(gridsearch.best_estimator_)
display(gridsearch.best_score_)
gridbest = gridsearch.best_estimator_

# %%
random_best.feature_importances_
#%%


# %%
evaluate_model(nb, X_train, X_test)
# %%
evaluate_model(rf, X_train, X_test)

# %%

feature_shape = rf.feature_importances_.shape[0]
# %%
rf.feature_importances_
# %%
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = vec_table.iloc[0][:feature_shape],
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances
# %%
feature_importances.reset_index(index)
# ! reser index to be ve_table columns
# rf.classes_
# %%
importances = rf.feature_importances_
np.argsort(importances)[::-1]

feature_names = vectorizer.get_feature_names()
top_words = []

for i in range(100):
    top_words.append(feature_names[indices[i]])
top_words
# %%
lr = LogisticRegression()
lr.fit(X_train, y_train)
# %%
# vec_table.index
lr.coef_.shape
# %%
def get_feature_coef(region_index):
    important_tokens = pd.DataFrame(data=lr.coef_[region_index],
                                    index=vectorizer.get_feature_names(),
                                    columns=['coefficient']).sort_values(by='coefficient', ascending=False)
    return important_tokens

# %%
east_token_coefs = get_feature_coef(0)
midwest_token_coefs = get_feature_coef(1)
south_token_coefs = get_feature_coef(2)
west_token_coefs = get_feature_coef(3)
print(east_token_coefs,midwest_token_coefs, south_token_coefs,west_token_coefs)
# %%
evaluate_model(lr, X_train, X_test)
# %%
# %%
df.lyrics = df.lyrics.apply(lambda x: x.split('  '))
# df.lyrics[1]
# %%
df.lyrics[1][7]

# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%