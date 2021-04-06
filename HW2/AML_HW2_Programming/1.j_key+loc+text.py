#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:36:03 2020

@author: zzhajun
"""
import pandas as pd
import re

from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

train=pd.read_csv("split_train.csv")
x_train=train.drop('target',axis=1)#dataframe with index
x_train=x_train.where(x_train.notnull(), '0')
x_train=x_train.values.tolist()
for i in range(5329):
    x_train[i][3]=str(x_train[i][1])+' '+str(x_train[i][2])+' '+x_train[i][3]
x_train=pd.DataFrame(x_train,columns=['id', 'keyword', 'location','text'])
y_train=train.target

test=pd.read_csv("dev.csv")
x_test=test.drop('target',axis=1)#dataframe with index
x_test=x_test.where(x_test.notnull(), '0')
x_test=x_test.values.tolist()
for i in range(2284):
    x_test[i][3]=str(x_test[i][1])+' '+str(x_test[i][2])+' '+x_test[i][3]
x_test=pd.DataFrame(x_test,columns=['id', 'keyword', 'location','text'])
y_test=test.target

tweettoken = TweetTokenizer(strip_handles=True, reduce_len=True)
lemmatizer=WordNetLemmatizer()

train_pre_new=[]
test_pre_new=[]
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
        train_pre_new.append(ret)
    elif dataclass==0:
        test_pre_new.append(ret)

def splitclass(data,q,m):
         for j in range(q):
                 preprocess(data["text"].iloc[j],m)
                 
splitclass(x_train,5329,1)
splitclass(x_test,2284,0)


train_result = pd.DataFrame(train_pre_new) 
train_result = train_result.join(y_train)
test_result = pd.DataFrame(test_pre_new) 
test_result = test_result.join(y_test)

train_result.to_csv('train_pre+key+loc.csv', sep=',',header=True, index=False)
test_result.to_csv('test_pre+key+loc.csv', sep=',',header=True, index=False)


# In[85]:
import numpy as np
import matplotlib.pyplot as plt

train=pd.read_csv("train_pre+key+loc.csv")
x_train=train.iloc[:, 0]
y_train=train.iloc[:, 1]

train=pd.read_csv("test_pre+key+loc.csv")
x_test=test.iloc[:, 0]
y_test=test.iloc[:, 1]


# In[86]:
# use bag of words model for text vectorization
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=5,binary=True) #set M=5 for svm
#vectorizer = CountVectorizer(min_df=5,binary=True) #set M=1 for lr and lsvm,binary=True) #instantiate the CountVectorizer class, set M=2

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


# In[89]:
# 1.f

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#LR model
LR = LogisticRegression()
LR.fit(train_vectors,y_train)
predict_lr=LR.predict(test_vectors)
print(f1_score(y_test,predict_lr))
#coefficient
conf_mat = confusion_matrix(y_test, predict_lr)
print(conf_mat)
print(classification_report(y_test, predict_lr))
coef_lr=LR.coef_[0]
maxindex_lr = np.argmax(coef_lr )
minindex_lr = np.argmin(coef_lr )

for i in train_voca_dic.keys():
    if train_voca_dic[i]==maxindex_lr:
        print('is real',i)
    if train_voca_dic[i]==minindex_lr:
        print('not real',i)
#0.7464940668824164
#is real: hiroshima  
#not real: full
# In[90]:
#1.g
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

Cs=[0.01, 0.1, 1.0, 10.0, 100.0]
f1_scores_l=[]

for C in Cs:
    LSVM=LinearSVC(C=C,max_iter=100000)
    LSVM.fit(train_vectors,y_train)
    predict_lsvm=LSVM.predict(test_vectors)
    f1_scores_l.append(f1_score(y_test,predict_lsvm))
    #print(LSVM.score(test_vectors,y_test))
    print(f1_score(y_test,predict_lsvm))

## plot
fig1=plt.figure()
ax=fig1.add_subplot(1,1,1)
ax.plot(Cs,f1_scores_l)
ax.set_xlabel(r"C")
ax.set_ylabel(r"f1_scores")
ax.set_xscale('log')
ax.set_title("LinearSVC")
plt.show()
#Linear SVM model
lsvm_model=LinearSVC(C=0.01)
lsvm_model.fit(train_vectors,y_train)
predict_lsvm=lsvm_model.predict(test_vectors)
print(f1_score(y_test,predict_lsvm))

#coefficient
coef_lsvm=lsvm_model.coef_[0]
maxindex_lsvm = np.argmax(coef_lsvm )
minindex_lsvm = np.argmin(coef_lsvm )
for i in train_voca_dic.keys():
    if train_voca_dic[i]==maxindex_lr:
        print('is real',i)
    if train_voca_dic[i]==minindex_lr:
        print('not real',i)
#0.7447795823665895
#is real: hiroshima  
#not real: full
# In[91]:
#1.h

from sklearn.svm import SVC
from sklearn.metrics import f1_score
'''
G=[0.01,0.1,1,5,10]
for g in G:
    SVM=SVC(C=1,kernel='rbf',gamma=g)
    SVM.fit(train_vectors,y_train)
    predict_svm=SVM.predict(test_vectors)
    print(f1_score(y_test,predict_svm))
    
G=[0.05,0.08,0.11,0.14,0.17]
for g in G:
    SVM=SVC(C=1,kernel='rbf',gamma=g)
    SVM.fit(train_vectors,y_train)
    predict_svm=SVM.predict(test_vectors)
    print(f1_score(y_test,predict_svm))


#best gamma=0.11
#best C=1
'''
Cs_svm=[0.01, 0.1, 1.0, 10.0, 100.0]
f1_scores_nl=[]

for c in Cs_svm:
    SVM=SVC(C=c,kernel='rbf',gamma=0.11,max_iter=100000)
    SVM.fit(train_vectors,y_train)
    predict_svm=SVM.predict(test_vectors)
    f1_scores_nl.append(f1_score(y_test,predict_svm))
    print(f1_score(y_test,predict_svm))


## plot
fig2=plt.figure()
ax=fig2.add_subplot(1,1,1)
ax.plot(Cs_svm,f1_scores_nl)
ax.set_xlabel(r"C")
ax.set_ylabel(r"f1_scores")
ax.set_xscale('log')
ax.set_title("NoneLinearSVC")
plt.show()
'''
#SVM model
SVM=SVC(C=1,kernel='rbf',gamma=0.11)
SVM.fit(train_vectors,y_train)
predict_svm=SVM.predict(test_vectors)
print(f1_score(y_test,predict_svm))

#0.7435897435897435
'''




















