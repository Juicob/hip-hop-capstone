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
# %%
bracket_text_to_remove = re.findall(reg_brackets, ','.join(corpus))
# %%
corpus = df.lyrics.to_list()
stopword_list = stopwords.words('english')
stopword_list.extend(string.punctuation)
stopword_list
# %%
tokens = word_tokenize(','.join(corpus))
freq = FreqDist(tokens)
freq.most_common(100)
# %%

# %%

# %%

# %%
