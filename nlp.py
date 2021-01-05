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
df = pd.read_csv('../capstone-data/lyrics/lyrics_main.csv',encoding='unicode_escape')

df.artist = df.artist.apply(lambda x: x.replace(x, str(x.split(' ')[:-1])).replace("['",'').replace("']",'').replace("', '", " ").replace('\\','') if x in artists_to_clean else x)
# %%
df.artist.value_counts()
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

# %%
