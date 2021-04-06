#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:40:38 2020

@author: zzhajun
"""

import pandas as pd
from sklearn.model_selection import train_test_split

train=pd.read_csv("train.csv")
x_train=train.drop('target',axis=1)#dataframe with index
y_train=train.target
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)
Train = X_train.join(y_train)
Train.to_csv('split_train.csv', sep=',',header=True, index=False)
Test = X_test.join(y_test)
Test.to_csv('dev.csv', sep=',',header=True, index=False)