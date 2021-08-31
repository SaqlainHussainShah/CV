import pandas as pd
import scipy.io
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import numpy as np

two_branch_features_path = "E:/saqlain/two_branch/wav_features_train.csv"
two_branch_test_path = "E:/saqlain/two_branch/wav_features_test.csv"


df = pd.read_csv(two_branch_features_path)
df_test =  pd.read_csv(two_branch_test_path)

X, Y =  df[df.columns[:-1]].values, df[df.columns[-1]].values
X_test, Y_test =  df_test[df_test.columns[:-1]].values, df_test[df_test.columns[-1]].values

le = preprocessing.LabelEncoder()
le.fit(Y)
labels= le.transform(Y)
X_train = X
y_train = labels
X_test = X_test
y_test = le.transform(Y_test)

rslt = []

clf = svm.SVC(decision_function_shape='ovo')
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, shuffle = True)
print("Starting training")
for i in range(10):
    print("{} th iteration ".format(i+1))
    clf.fit(X_train, y_train)
    
    
    
    test_data =  np.array(X_test)
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1])
    
    y_test = y_test.reshape(y_test.shape[0],1)
    print("starting prediction")
    predictions = clf.predict(test_data)
    rslt.append(accuracy_score(predictions, y_test))
    print("Accuracy : {}".format(rslt[-1]))
    
print(np.mean(rslt))
    
# from sklearn.metrics import classification_report

# print(classification_report(y_test, predictions))  #Inputs for fitting were not label encoded