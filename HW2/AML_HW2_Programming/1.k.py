#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:52:24 2020

@author: zzhajun
"""

import pandas as pd
import re

from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

train=pd.read_csv("train.csv")
x_train=train.drop('target',axis=1)#dataframe with index
y_train=train.target

x_test=pd.read_csv("test.csv")


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
                 
splitclass(x_train,7613,1)
splitclass(x_test,3263,0)


train_result = pd.DataFrame(train_pre) 
train_result = train_result.join(y_train)
test_result = pd.DataFrame(test_pre) 


train_result.to_csv('train_pre_total.csv', sep=',',header=True, index=False)
test_result.to_csv('test_pre_total.csv', sep=',',header=True, index=False)

#train=pd.read_csv("train_pre_total.csv")
x_train=train_result.iloc[:, 0]
y_train=train_result.iloc[:, 1]
x_test=test_result.iloc[:,0]

# In[86]:
# use bag of words model for text vectorization
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=3,binary=True) #instantiate the CountVectorizer class, set M=3

train_corpus = x_train
print (train_corpus.head(5))

train_vectors = vectorizer.fit_transform(train_corpus) #transform each document into a word frequency vector
print(type(train_vectors)) #type: sparse vector
print (train_vectors)

train_vectors = train_vectors.toarray()
print (train_vectors)

train_voca_list = vectorizer.get_feature_names() #generate corpus into a vocabulary list
train_voca_dic = vectorizer.vocabulary_
print('vocabulary list of trainset:', train_voca_list)
print( 'vocabulary dic of trainset:', train_voca_dic)

print(type(train_vectors)) #type: np array
print(train_vectors.sum(axis=0)) #count each word' frequency in the corpus

X1 = train_vectors.shape
print (X1)
train_feature_num = X1[1] #vector length/number of features
print ('total number of features in trainset:', train_feature_num)


# In[88]:

test_corpus = x_test
print (test_corpus.head(5))

test_vectors = vectorizer.transform(test_corpus) #transform each document into a word frequency vector
print(type(test_vectors)) #type: sparse vector
print (test_vectors)

test_vectors = test_vectors.toarray()
print (test_vectors)

test_voca_list = vectorizer.get_feature_names() #generate corpus into a vocabulary list
test_voca_dic = vectorizer.vocabulary_
print('vocabulary list of trainset:', test_voca_list)
print( 'vocabulary dic of trainset:', test_voca_dic)

print(type(test_vectors)) #type: np array
print(test_vectors.sum(axis=0)) #count each word' frequency in the corpus

X2 = test_vectors.shape
print (X2)
test_feature_num = X2[1] #vector length/number of features
print ('total number of features in devset:', test_feature_num)


# In[16]:


# handwrite a BernoulliNB model


import numpy as np

class Bernoulli_NaiveBayes:

    def __init__(self):   
        self.alpha = 1 # set smoothing factor=1(Laplace Smoothing), to avoid zero probability problems  

    def _cal_prior_prob_log(self, y, classes): # calculate the logarithm of prior probability of each class, P(y=c_k)
        self.classes = np.unique(y)
        class_num = len(self.classes) #count the number of possible types of y
        sample_num = len(y)
        
        c_num = np.count_nonzero(y == classes[:, None], axis=1) #count sample amount of each class
        prior_prob = (c_num + self.alpha) / (sample_num + class_num * self.alpha) #calculate prior probabilities(add smoothing correction)
        prior_prob_log = np.log(prior_prob) #calculate logarithm
        
        return prior_prob_log
    
    def _cal_condi_prob_log(self, X, y, classes): #calculate the logarithm of all conditional probabilities P(x^(j)|y=c_k)
        
        n = (X.shape)[1]
        K = len(classes)
        
        #create an empty multidimensional array
        #prob_log: logarithmic matrix of two conditional probabilities
        condi_prob_log = np.empty((2, K, n)) 
        
        for k, c in enumerate(classes):
            X_c = X[np.equal(y, c)] #acquire all samples of class c_k
            total_num = len(X_c)
            num_f1 = np.count_nonzero(X_c, axis=0) #count the number of samples of which feature value is 1
            condi_prob_f1 = (num_f1 + self.alpha) / (total_num + self.alpha * 2) #calculate conditional probability P(x^(j)=1|y=c_k)
            
            #calculate and store logarithm into matrix
            #prob_log[0]: store all values of log(P(x^(j)=0|y=c_k))
            #prob_log[1]: store all values of log(P(x^(j)=1|y=c_k))
            condi_prob_log[0, k] = np.log(1 - condi_prob_f1) 
            condi_prob_log[1, k] = np.log(condi_prob_f1) 
            
        return condi_prob_log
   
    def train(self, x_train, y_train): #train the model
        self.classes = np.unique(y_train) #acquire all classes  
        self.prior_prob_log = self._cal_prior_prob_log(y_train, self.classes) #calculate and store the logarithm of all prior probabilities
        self.condi_prob_log = self._cal_condi_prob_log(x_train, y_train, self.classes) #calculate and store the logarithm of all conditional probabilities

    def _predict_single_sample(self, x): #predict the label of single sample

        K = len(self.classes)
        po_prob_log = np.empty(K) #create an empty multidimensional array
        
        index_f1 = x == 1 #acquire index of feature value=1 
        index_f0 = ~index_f1 #acquire index of feature value=0

        for k in range(K): #iterate each class
            #calculate the logarithm of the numerator of the posterior probability
            po_prob_log[k] = self.prior_prob_log[k]                                 + np.sum(self.condi_prob_log[0, k][index_f0])                                 + np.sum(self.condi_prob_log[1, k][index_f1])

        label = np.argmax(po_prob_log) #get the class with the highest posterior probability
        return label

    def predict(self, X): #predict samples (include single sample)
        
        if X.ndim == 1: #if only predict single sample (the dimension of the array = 1), invoke _predict_single_sample()
            return self._predict_single_sample(X) 
        else:
            #if predict multiple samples, loop call _predict_single_sample() and return a list of the predicted results 
            labels = []
            for j in range(X.shape[0]):
                label = self._predict_single_sample(X[j])
                labels.append(label)
            return labels

# In[17]:


#use Bernoulli_NaiveBayes to predict 



x_train = train_vectors
y_train = np.array(y_train)
x_test = test_vectors

BernoulliNB = Bernoulli_NaiveBayes()
BernoulliNB.train(x_train,y_train)
y_pred = BernoulliNB.predict(x_test)
result=pd.DataFrame(y_pred,columns=['target'])
x_test=pd.read_csv("test.csv")
test_id=x_test.iloc[:,0]
test_id=test_id.to_frame()
result=test_id.join(result)
result.to_csv('prediction.csv', sep=',',header=True, index=False)





















