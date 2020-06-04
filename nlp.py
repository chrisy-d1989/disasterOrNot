# -*- coding: utf-8 -*-
"""

modul: help functions
modul author: Christoph Doerr

https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')

import re
from nltk.tokenize import word_tokenize
import string
from keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer 
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from nltk.stem.porter import PorterStemmer
import utils as utils

nlp_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/nlp-getting-started/'

train_raw = pd.read_csv(nlp_path +'train.csv')
test_raw = pd.read_csv(nlp_path + 'test.csv')
tqdm.pandas()
if os.path.exists('{}{}.csv'.format(nlp_path, 'train_clean')) and \
    os.path.exists('{}{}.csv'.format(nlp_path, 'test_clean')):
    print('Loading data from file ...')
    train_clean = pd.read_csv(nlp_path +'train_clean.csv')
    test_clean = pd.read_csv(nlp_path + 'test_clean.csv')
    print('... done loading data from file !!!')
else:
    train = pd.read_csv('C:/Users/cdoerr1/Desktop/CoronaAi/nlp-getting-started/train.csv')
    test = pd.read_csv('C:/Users/cdoerr1/Desktop/CoronaAi/nlp-getting-started/test.csv')
    train_clean = train['text'].progress_apply(lambda x : utils.cleanTweet(x, appostrophes=True, emojis=True, html=True, url=True,\
                                                                     misspellings=False, punctuation=True, lemming=True, stop=True))
    utils.safeIndicators(train_clean, nlp_path, 'train_clean')
    print('finished cleanning train dataset')
    test_clean = train['text'].progress_apply(lambda x : utils.cleanTweet(x, appostrophes=True, emojis=True, html=True, url=True,\
                                                                     misspellings=False, punctuation=True, lemming=True, stop=True))
    print('finished cleanning train dataset')
    utils.safeIndicators(test_clean, nlp_path, 'test_clean')

data = pd.concat([train_clean,test_clean])
data_corpus = utils.create_corpus(data)
embedding_dict={}
with open('C:/Users/cdoerr1/Desktop/CoronaAi/nlp-getting-started/glove.6B.100d.txt','r', encoding="utf8") as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()

MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(data_corpus)
sequences=tokenizer_obj.texts_to_sequences(data_corpus)
tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))
for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue   
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
            
model=Sequential()
embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=1e-5),metrics=['accuracy'])
train=tweet_pad[:train_raw.shape[0]]
test=tweet_pad[train_raw.shape[0]:]

X_train,X_test,y_train,y_test=train_test_split(train,train_raw['target'].values,test_size=0.15)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)

