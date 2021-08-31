# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:41:57 2021

@author: Muhammad Saad Saeed (18F-MS-CP-01)
"""



"""Resulted in 68% accuracy on 8251 test data"""

import pandas as pd
import scipy.io
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score

two_branch_features_path = "E:/saqlain/two_branch/wav_features_test.csv"

df = pd.read_csv(two_branch_features_path)

X, Y =  df[df.columns[:-1]].values, df[df.columns[-1]].values


le = preprocessing.LabelEncoder()
le.fit(Y)
labels= le.transform(Y)

rslt = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, shuffle = True)
    
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    
    
    
    test_data =  np.array(X_test)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])
    
    y_test = y_test.reshape(y_test.shape[0],1)
    predictions = clf.predict(test_data)
    
    rslt.append(accuracy_score(predictions, y_test))
    # print(classification_report(y_test, predictions))
    print(i)

"""Resulted in 68% accuracy on 8251 test data"""
