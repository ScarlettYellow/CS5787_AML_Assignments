#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:26:27 2020

@author: zzhajun
"""

import pandas as pd
import re

from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

train=pd.read_csv("split_train.csv")
x_train=train.drop('target',axis=1)#dataframe with index
y_train=train.target

test=pd.read_csv("dev.csv")
x_test=test.drop('target',axis=1)#dataframe with index
y_test=test.target

tweettoken = TweetTokenizer(strip_handles=True, reduce_len=True)
lemmatizer=WordNetLemmatizer()


train_pre=[]
test_pre=[]
data=[]
def preprocess(text,dataclass):
    url = re.compile(r'https?://\S+|www\.\S+')#delete url
    text_new=url.sub(r'',text)
    text_new=re.sub('[^a-zA-Z]'," ",text_new)#delete punctuation
    text_new=text_new.lower()#lowercase
    text_rest=tweettoken.tokenize(text_new)
    for i in text_rest:
        if i in stopwords.words('english'):#delete stopwords
            text_rest.remove(i)
    rest=[]
    for k in text_rest:
        rest.append(lemmatizer.lemmatize(k))#lemmatize
    ret=" ".join(rest)
    if dataclass==1:
        train_pre.append(ret)
    elif dataclass==0:
        test_pre.append(ret)

def splitclass(data,q,m):
         for j in range(q):
                 preprocess(data["text"].iloc[j],m)
                 
splitclass(x_train,5329,1)
splitclass(x_test,2284,0)


train_result = pd.DataFrame(train_pre) 
train_result = train_result.join(y_train)
test_result = pd.DataFrame(test_pre) 
test_result = test_result.join(y_test)

train_result.to_csv('train_pre.csv', sep=',',header=True, index=False)
test_result.to_csv('test_pre.csv', sep=',',header=True, index=False)

























