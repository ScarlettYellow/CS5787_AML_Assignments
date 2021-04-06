#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:53:43 2020

@author: zzhajun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import make_scorer, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

#Problem 1.a
train=pd.read_csv("train.csv")
x_test_image=pd.read_csv("test.csv")

#Problem 1.b
x_train_image=train.drop('label',axis=1)#dataframe with index
y_train_label=train.label

X_train=x_train_image.values.reshape(42000,28,28).astype('float32')
X_test=x_test_image.values.reshape(28000,28,28).astype('float32')

x_train_data = np.array(x_train_image)#array 42000*784

images=[]
labels=[]
num=[1,0,16,7,3,8,21,6,10,11]
for i in range(10):
    images.append(X_train[num[i]])
    labels.append(y_train_label[num[i]])

def plot_images_labels(images,labels,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)
    for i in range(0,10):
        ax=plt.subplot(5,5,i+1)
        ax.imshow(images[idx],cmap='binary')
        title='label='+str(labels[idx])
        
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
    
plot_images_labels(images,labels,idx=0)

#Problem 1.c
count=np.zeros((10,), dtype=np.int)
prob=np.zeros((10,))
for i in range(len(y_train_label)):
    for j in range(10):
        if y_train_label[i]==j:
            count[j]+=1
print(count)

for i in range(10):
    prob[i]=count[i]/42000
    print(prob[i])
plt.hist(x = y_train_label, density=True, bins = 10, color = 'steelblue', edgecolor = 'black' )

plt.xlabel('number')
plt.ylabel('probability')
plt.title('prior probability')
plt.show()          


#Problem 1d
a=0
dist=[10000,10000,10000,10000,10000,10000,10000,10000,10000,10000]
match=np.zeros(10)
for i in range(10):
    for j in range(42000):
        if j!=num[i]:
            a=np.linalg.norm(images[i]-X_train[j])
            if dist[i]>a:
                dist[i]=a
                match[i]=y_train_label[j]

print(dist)
print(match)


#Problem 1.e
num0=[]
image0=[]
total0=0
num1=[]
image1=[]
total1=0
for i in range(42000):
    if y_train_label[i]==0:
        num0.append(y_train_label[i])
        image0.append(x_train_data[i])
        total0+=1
    elif y_train_label[i]==1:
        num1.append(y_train_label[i])
        image1.append(x_train_data[i])
        total1+=1
#print(total0,total1)
dist00=cdist(image0,image0,'euclidean')
dist11=cdist(image1,image1,'euclidean')
dist01=cdist(image0,image1,'euclidean')

dist00=dist00.reshape(1,4132*4132)
dist11=dist11.reshape(1,4684*4684)
dist01=dist01.reshape(1,4132*4684)

dist00=np.array(dist00)
dist00=dist00.tolist()

dist11=np.array(dist11)
dist11=dist11.tolist()

dist01=np.array(dist01)
dist01=dist01.tolist()

dist_genu=dist00[0]+dist11[0]
dist_impo=dist01[0]+dist01[0]

sns.distplot(dist_genu, bins = 100, kde = False, hist_kws = {'color':'steelblue'}, label = 'genuine')
sns.distplot(dist_impo, bins = 100, kde = False, hist_kws = {'color':'purple'}, label = 'impostor')
plt.title('distance')
plt.legend()
plt.show()

#Problem 1.f

genu_min=min(dist_genu)
genu_max=max(dist_genu)
impo_min=min(dist_impo)
impo_max=max(dist_impo)
dist_max=max(genu_max,impo_max)
distmax=[]
distmax.append(dist_max)
print(dist_max)#max distance

inter=60
count_tp=np.zeros(60)
count_fp=np.zeros(60)

#TPR
for j in range(60):
    for i in range(39013280): 
        if dist_genu[i]<distmax[0]-inter*j:
            count_tp[j]+=1
count_tp=count_tp/ 39013280

#FPr      
for j in range(60):
    for i in range(38708576): 
        if dist_impo[i]<distmax[0]-inter*j:
            count_fp[j]+=1
count_fp=count_fp/ 38708576
e=0.001
for i in range(60):
    if (abs(count_tp[i]+count_fp[i]-1)<e):
        print('ERR=',count_tp[i])

plt.plot(count_fp,count_tp,label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()






