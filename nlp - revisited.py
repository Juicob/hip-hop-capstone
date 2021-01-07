# %%
import os
import sys
import datetime
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
from gensim.models import word2vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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


# %%
def clean_lyrics(lyrics):
    
    ## Convert words to lower case and split them
    lyrics = lyrics.lower().split()
    
    ## Remove stop words
    stopwords_list = set(stopwords.words("english"))
    lyrics = [w for w in lyrics if not w in stopwords_list]
    # stopword_list = stopwords.words('english')
    # stopword_list += string.punctuation
    
    lyrics = " ".join(lyrics)
    ## Clean the lyrics
    lyrics = re.sub(r"\[[^\]]*\]", " ", lyrics)
    lyrics = re.sub(r"\n", " ", lyrics)
    lyrics = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", lyrics)
    return lyrics

# %%
df.artist = df.artist.apply(lambda x: x.replace(x, str(x.split(' ')[:-1])).replace("['",'').replace("']",'').replace("', '", " ").replace('\\','') if x in artists_to_clean else x)

# %%
# reg_brackets = '\[[^\]]*\]'
# reg_newline = '\n'
# df.lyrics[1][400:500]
# df.lyrics.to_list()
# re.sub(reg_brackets, ' ', ','.join(corpus))
# re.sub(reg_newline, ' ', corpus)
# %%
# df.lyrics = df.lyrics.apply(lambda x: re.sub('\[[^\]]*\]', ' ', x))
df.lyrics = df.lyrics.apply(lambda x: clean_lyrics(x))

# %%
df.lyrics[1]
# %%
y = df['region_code']
X = df['lyrics']
# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=123) 
X_train.shape,y_test.shape
# %%
weights= compute_class_weight(
           'balanced',
            np.unique(y_train), 
            y_train)

weights_dict = dict(zip( np.unique(y_train),weights))
weights_dict
# %%
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# %%
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# %%

# %%

# %%

# %%
# %%
y_train
y_test
# %%
## TOKENIZE TEXT
MAX_WORDS = 6000
MAX_SEQUENCE_LENGTH = 50

tokenizer = text.Tokenizer(num_words=MAX_WORDS)

tokenizer.fit_on_texts(X_train) #df['text'])
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)#df['text'])


## Find the longest sequence
seq_lenghts = list(map(lambda x: len(x),[*y_train,*y_test]))
max(seq_lenghts)
# %%
## Pad sequences
X_train = sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

X_test = sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_train
# %%
len(tokenizer.index_word)
# %%
# %%
# %%
def get_earlystop(monitor='val_loss',patience=5, restore_best_weights=False):
    """"""
    args = locals()
    return EarlyStopping(**args)

get_earlystop.__doc__+=EarlyStopping.__doc__
# %%
model=Sequential()

model.add(Embedding(MAX_WORDS, 128))
model.add(LSTM(50,return_sequences=False))
#     model.add(GlobalMaxPool1D()) 
#     model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',#'categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
display(model.summary())

history = model.fit(X_train, y_train, epochs=50,
                batch_size=32, validation_split=0.2,callbacks=get_earlystop(),
                class_weight=weights_dict)
plot_results(history.history)
evaluate_results(model)

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

# %%

# %%

# %%

# %%

# %%
total_vocabulary = set(word for lyrics in data for word in lyrics)
# %%
len(total_vocabulary)
print('There are {} unique tokens in the dataset.'.format(len(total_vocabulary)))
# %%
glove = {}
with open('glove.6B.50d.txt', 'rb') as f:
    for line in f:
        parts = line.split()
        word = parts[0].decode('utf-8')
        if word in total_vocabulary:
            vector = np.array(parts[1:], dtype=np.float32)
            glove[word] = vector
# %%
glove['game']
# %%
class W2vVectorizer(object):
    
    def __init__(self, w2v):
        # Takes in a dictionary of words and vectors as input
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(glove))])
    
    # Note: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # it can't be used in a scikit-learn pipeline  
    def fit(self, X, y):
        return self
            
    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                   or [np.zeros(self.dimensions)], axis=0) for words in X])
# %%
rf =  Pipeline([('Word2Vec Vectorizer', W2vVectorizer(glove)),
              ('Random Forest', RandomForestClassifier(n_estimators=100, verbose=True))])
svc = Pipeline([('Word2Vec Vectorizer', W2vVectorizer(glove)),
                ('Support Vector Machine', SVC())])
lr = Pipeline([('Word2Vec Vectorizer', W2vVectorizer(glove)),
              ('Logistic Regression', LogisticRegression())])
# %%
models = [('Random Forest', rf),
          ('Support Vector Machine', svc),
          ('Logistic Regression', lr)]
# %%
scores = [(name, cross_val_score(model, data, target, cv=2).mean()) for name, model, in models]
# %%
scores
# %%
y = pd.get_dummies(target).values
# %%
tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(df['lyrics']))
list_tokenized_lyrics = tokenizer.texts_to_sequences(df['lyrics'])
X_test = sequence.pad_sequences(list_tokenized_lyrics, maxlen=5000)
# %%
model = Sequential()
embedding_size = 128
model.add(Embedding(20000, embedding_size))
model.add(LSTM(25, return_sequences=False))
# model.add(GlobalMaxPool1D())
model.add(Dropout(0.5))
# model.add(LSTM(25, return_sequences=True))
# model.add(GlobalMaxPool1D())
# model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

history1 = model.fit(X_test, y, epochs=5, batch_size=32, validation_split=0.5)
plot_results(history1.history)
# %%
# %%
# evaluate_results(model)
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
reg_brackets = '\[[^\]]*\]'
reg_newline = '\n'
df.lyrics[1][400:500]
corpus = df.lyrics.to_list()
corpus = re.sub(reg_brackets, ' ', ','.join(corpus))
corpus = re.sub(reg_newline, ' ', corpus)

# %%
corpus[:3000]

# %%
# %%
stopword_list = stopwords.words('english')
stopword_list += string.punctuation
stopword_list[-20:]
# %%
tokens = word_tokenize(corpus)
# %%
len(tokens)
# %%
cleaned_tokens = [w.lower() for w in tokens[:] if w.lower() not in stopword_list]
# %%
freq = FreqDist(cleaned_tokens)
freq.most_common(100)

# %%

# %%
df = pd.read_csv('../capstone-data/lyrics/lyrics_main.csv',encoding='unicode_escape')

# %%
df['cleaned_lyrics'] = df.lyrics.copy(deep=True)
# %%
df.cleaned_lyrics = df.cleaned_lyrics.apply(lambda x: re.sub(reg_brackets, '', x))
df.cleaned_lyrics = df.cleaned_lyrics.apply(lambda x: re.sub(reg_newline, ' ', x))
# %%
df.cleaned_lyrics.head()
# %%

# %%
# ? break

# %%

df = pd.read_csv('../capstone-data/lyrics/lyrics_main.csv',encoding='unicode_escape')
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


# %%
df.artist = df.artist.apply(lambda x: x.replace(x, str(x.split(' ')[:-1])).replace("['",'').replace("']",'').replace("', '", " ").replace('\\','') if x in artists_to_clean else x)
# %%
df.artist.value_counts()
# %%
df = df.groupby('region').filter(lambda x : len(x) > 1)
df = df.groupby('artist').filter(lambda x : len(x) > 1)
df.region = pd.Categorical(df.region)
df.artist = pd.Categorical(df.artist)
df.region = df.region.cat.codes
df.artist = df.artist.cat.codes

# %%
df['lyrics_length'] = df.lyrics.apply(lambda x: len(x))
# %%
df.lyrics_length.describe()
# %%
df[df.lyrics_length > 10000]
# ! remove credits and other outliers
# %%
df, df_test = train_test_split(df, test_size = 0.2, stratify = df[['region']])
# %%
# Name of the BERT model to use
model_name = 'bert-base-uncased'
# Max length of tokens
max_length = 6000
# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False
# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config)
# %%
# Load the MainLayer
bert = transformer_model.layers[0]
# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
inputs = {'input_ids': input_ids}
# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
# Then build your model output
region = Dense(units=len(df.region.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='region')(pooled_output)
artist = Dense(units=len(df.artist.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='artist')(pooled_output)

outputs = {'region': region,'artist': artist}
# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
# Take a look at the model
model.summary()
# %%
# Tokenize the input (takes some time)
x = tokenizer(
    text=df['lyrics'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)
# %%
# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss = {'region': CategoricalCrossentropy(from_logits = True), 'artist': CategoricalCrossentropy(from_logits = True)}
metric = {'region': CategoricalAccuracy('accuracy'), 'artist': CategoricalAccuracy('accuracy')}
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)
# Ready output data for the model
y_region = to_categorical(df.region)
y_artist = to_categorical(df.artist)


# Fit the model
history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'region': y_region, 'artist': y_artist},
    validation_split=0.2,
    batch_size=64,
    epochs=10)
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
df.release_date.value_counts()

# ? check complaints csv and article
#%%
data = pd.read_csv('D:/Downloads2/complaints.csv')

#%%
data = data.dropna()

# Remove rows, where the label is present only ones (can't be split)
data = data.groupby('Issue').filter(lambda x : len(x) > 1)
data = data.groupby('Product').filter(lambda x : len(x) > 1)
# %%
data.head()
# %%
# Set your model output as categorical and save in new label col
data['Issue_label'] = pd.Categorical(data['Issue'])
data['Product_label'] = pd.Categorical(data['Product'])
# %%
# Transform your output to numeric
data['Issue'] = data['Issue_label'].cat.codes
data['Product'] = data['Product_label'].cat.codes
# %%
data.head()
data.shape
# %%
# Split into train and test - stratify over Issue
data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['Issue']])
# %%
#######################################
### --------- Setup BERT ---------- ###

# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 100

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config)


#######################################
### ------- Build the model ------- ###

# TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model

# Load the MainLayer
bert = transformer_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
# attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
# inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
issue = Dense(units=len(data.Issue_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='issue')(pooled_output)
product = Dense(units=len(data.Product_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='product')(pooled_output)
outputs = {'issue': issue, 'product': product}

# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# Take a look at the model
model.summary()


#######################################
### ------- Train the model ------- ###

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'issue': CategoricalCrossentropy(from_logits = True), 'product': CategoricalCrossentropy(from_logits = True)}
metric = {'issue': CategoricalAccuracy('accuracy'), 'product': CategoricalAccuracy('accuracy')}

# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

# Ready output data for the model
y_issue = to_categorical(data['Issue'])
y_product = to_categorical(data['Product'])

# Tokenize the input (takes some time)
x = tokenizer(
    text=data['Consumer complaint narrative'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

# Fit the model
history = model.fit(
    # x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
    x={'input_ids': x['input_ids']},
    y={'issue': y_issue, 'product': y_product},
    validation_split=0.2,
    batch_size=32,
    verbose=1,
    epochs=10)

# %%

# %%
