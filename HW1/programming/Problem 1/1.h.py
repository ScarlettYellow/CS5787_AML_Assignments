#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:23:35 2020

@author: zzhajun
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
   
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__=='__main__':
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
     
    errorCount = 0.0
   
    mTest = 6300
    pred=[]
    true=[]
    for i in range(mTest):
        true.append(y_test[i])
    
    for k in range(2,10):
        neigh =KNN(n_neighbors =k, algorithm = 'auto')
   
        neigh.fit(X_train_normalize, y_train)
        for i in range(mTest):
            classifierResult = neigh.predict([X_test_normalize[i]])
            pred.append(classifierResult)
        #print("predict%d\tlabel%d" % (classifierResult, y_test[i]))
            if(classifierResult != y_test[i]):
                errorCount += 1.0
        acc=(1-errorCount/mTest)*100     
        print(k, acc,'%')
   
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(true, pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print (cm_normalized)
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
