#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:06:18 2020

@author: zzhajun
"""
import operator
import numpy as np
import pandas as pd
#import time
from pandas.core.frame import DataFrame

def classify(inX,dataset,labels,k):
    #start=time.time()
    datasetsize = dataset.shape[0]
    ###calculate distance
    diffMat = np.tile(inX,(datasetsize,1))-dataset
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

if __name__ == '__main__':
    train=pd.read_csv("train.csv")
    test=pd.read_csv("test.csv")

    x_train_image=train.drop('label',axis=1)#dataframe with index
    y_train_label=train.label
    
    x_train_image=x_train_image.values
    y_train_label=y_train_label.values
    X_train_normalize=x_train_image/255
    
    x_test=test.values
    X_test_normalize=x_test/255
    
    Id=[]
    for i in range(1,28001):
        Id.append(i)
    Result=[]
    errorCount=0
    for i in range(28000):
        Result.append(classify( X_test_normalize[i], X_train_normalize,y_train_label, 3))
        #print(i, Result[i])
        
    ex = {"ImageId":Id,
            "Label":Result}
    d=DataFrame(ex)
    d.to_csv("./knn.csv", index=False)