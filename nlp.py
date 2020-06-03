# -*- coding: utf-8 -*-
"""

modul: help functions
modul author: Christoph Doerr

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
from spellchecker import SpellChecker
from nltk.stem.porter import PorterStemmer
import utils as utils

nlp_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/nlp-getting-started/'
# def create_corpus(series, target):
#     corpus=[]
    
#     for x in series[series['target']==target]['text'].str.split():
#         for i in x:
#             corpus.append(i)
#     return corpus



def findStopwords(corpus):
    dic_stopwords=defaultdict(int)
    dic_punctuation=defaultdict(int)
    stop_words=set(stopwords.words('english'))
    special = string.punctuation
    for word in corpus:
        if word in stop_words:
            dic_stopwords[word]+=1
        if word in special:
            dic_punctuation[word]+=1
    top=sorted(dic_stopwords.items(), key=lambda x:x[1],reverse=True)[:10] 
    return top, dic_punctuation

def get_top_tweet_bigrams(corpus, dim=2, n=None):
    vec = CountVectorizer(ngram_range=(dim, dim)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def cleanTweet(text, appostrophes=True, emojis=True, html=True, url=True, misspellings=True, punctuation=True, lemming=True,\
               stop=True):
    # for text in corpus:
    if appostrophes:
        #convert appostrophes
        filtered_tweet = utils.decontracted(text)
    if emojis:
        #decoding, removing emojis
        filtered_tweet = filtered_tweet.encode("utf-8").decode('ascii','ignore')
    if html:
        #cleaning of html tags
        htmltags = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        filtered_tweet = re.sub(htmltags, '', filtered_tweet)
    if url:
        #cleaning of url
        url = re.compile(r'https?://\S+|www\.\S+')
        filtered_tweet = re.sub(url, '', text)
    if misspellings:
        #cleaning of misspellings
        spell = SpellChecker()
        corrected_text = []
        misspelled_words = spell.unknown(filtered_tweet.split())
        for word in filtered_tweet.split():
            if word in misspelled_words:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)
        filtered_tweet =  " ".join(corrected_text)
    #cleaning of slang words
    #split attached words, not working and questionable because of all capital words
    # filtered_tweet =  " ".join(re.findall('[A-Z][^A-Z]*', filtered_tweet))
    #stemming
    if punctuation:
        word_tokens = word_tokenize(filtered_tweet)
        #remove punctuations
        table=str.maketrans('','',string.punctuation)
        filtered_tweet.translate(table)  
        filtered_tweet = [word.translate(table) for word in word_tokens]
        filtered_tweet = " ".join(filtered_tweet)
    if lemming:
        #lemming of words
        word_tokens = word_tokenize(filtered_tweet)
        lemmatizer = WordNetLemmatizer() 
        filtered_tweet = [lemmatizer.lemmatize(word) for word in word_tokens]
    if stop:
        # cleaning from stopwords
        stop_words=set(stopwords.words('english'))
        stop_word_drop = [] 
        for word in filtered_tweet: 
            if word not in stop_words: 
                stop_word_drop.append(word) 
    filtered_tweet = " ".join(stop_word_drop)
    return filtered_tweet

def create_corpus(df):
    corpus=[]
    stop_words=set(stopwords.words('english'))
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop_words))]
        corpus.append(words)
    return corpus

#cleaning of rare words
# tokens is a list of all tokens in corpus
# freq_dist = nltk.FreqDist(token)
# rarewords = freq_dist.keys()[-50:]
# after_rare_words = [ word for word in token not in rarewords]

#  lower case
# train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))   

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
    train_clean = train['text'].progress_apply(lambda x : cleanTweet(x, appostrophes=True, emojis=True, html=True, url=True,\
                                                                     misspellings=False, punctuation=True, lemming=True, stop=True))
    utils.safeIndicators(train_clean, nlp_path, 'train_clean')
    print('finished cleanning train dataset')
    test_clean = train['text'].progress_apply(lambda x : cleanTweet(x, appostrophes=True, emojis=True, html=True, url=True,\
                                                                     misspellings=False, punctuation=True, lemming=True, stop=True))
    print('finished cleanning train dataset')
    utils.safeIndicators(test_clean, nlp_path, 'test_clean')

data = pd.concat([train_clean,test_clean])
data_corpus = create_corpus(data)
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
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)

