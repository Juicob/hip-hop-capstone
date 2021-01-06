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
# import glob

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.collocations import *
from nltk import FreqDist
from nltk import word_tokenize
from textblob import TextBlob

import tensorflow as tf



# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
df.region = pd.Categorical(df.region)
df.region = df.region.cat.codes

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
input_ids = Input(shape=(max_length,), name='input_ids', dtype='tf.float32')
inputs = {'input_ids': input_ids}
# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)
# Then build your model output
region = Dense(units=len(df.region.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='region')(pooled_output)

outputs = {'region': region}
# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')
# Take a look at the model
model.summary()
# %%
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
y_region = to_categorical(df.region)
# Tokenize the input (takes some time)
x = tokenizer(
    text=df.lyrics.to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)
# Fit the model
history = model.fit(
    x={'input_ids': x['input_ids']},
    y={'region': y_region},
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
