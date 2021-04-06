#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 06:57:48 2020

@author: zzhajun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 06:03:13 2020

@author: zzhajun
"""

import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def classify(inX,dataset,labels,k):
    #start=time.time()
    diffMat = inX[None,:] - dataset
    #print(time.time()-start)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #print(time.time()-start)
    #sort distance, return index
    sortedDistIndicies = distances.argsort()
    #print(time.time()-start)
    #dictionary
    classCount = {}
    #k least distance
    for i in range(k):
        #sortedDistIndicies[0]index of min dist
        #labels[sortedDistIndicies[0]]label of min dist
        voteIlabel = labels[sortedDistIndicies[i]]
        #label as key,support key +1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #sort
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print(time.time()-start)
    return sortedClassCount[0][0]

def classify2(inX,dataset,labels,k):
    diffMat = inX[None,:] - dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

    
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
if __name__ == '__main__':
    train=pd.read_csv("train.csv")
    x_train_image=train.drop('label',axis=1)#dataframe with index
    y_train_label=train.label

    X_train, X_test, y_train, y_test = train_test_split(x_train_image, y_train_label, test_size=0.15, random_state=42)
    
    
    
    X_train_normalize=X_train/255
    X_test_normalize=X_test/255

    X_train_normalize=X_train_normalize.values
    X_test_normalize=X_test_normalize.values
    y_train=y_train.values
    y_test=y_test.values
    
    mTest = 6300
    mTrain=35700
    true_hold=[]
    true_train=[]
    for i in range(mTest):
        true_hold.append(y_test[i])
     
    for i in range(mTrain):
        true_train.append(y_train[i])
    Result_hold=[]
    Result_train=[]
    errorCount=0
    errorCount_train=0 
    
    for i in range(mTest):
        Result_hold.append(classify( X_test_normalize[i], X_train_normalize,y_train, 3))
        if (Result_hold[i]!= y_test[i]):
            errorCount+=1
        #print( Result_hold[i],y_test[i])
    acc_hold=(1-errorCount/mTest)*100     
    print(acc_hold,'%')
    
    
    for i in range(mTrain):
        Result_train.append(classify2(X_train_normalize[i], X_train_normalize,y_train, 3))
        if (Result_train[i]!= y_train[i]):
            errorCount_train+=1
        print(Result_train[i],y_train[i])
    acc_train=(1-errorCount_train/mTrain)*100     
    print(acc_train,'%')
    
    
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    tick_marks = np.array(range(len(labels))) + 0.5
    #cm = confusion_matrix(true_hold, Result_hold)
    cm1 = confusion_matrix(true_train, Result_train)
    np.set_printoptions(precision=2)
    #cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm1_normalized = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    print (cm1_normalized)
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm1_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(cm1_normalized, title='Normalized confusion matrix')
