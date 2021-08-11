# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:00:26 2021

@author: Muhammad Saad Saeed (18F-MS-CP-01)
"""


from glob import glob
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os

def loadSplit():
    train = []
    test = []
    
    with open('D:/saqlain/iden_split.txt','r+') as f:
        for dat in f:
            dat = dat.split('\n')[0]
            dat = dat.split(' ')
            if dat[0] =='1' or dat[0] == '2':
                train.append(dat[1])
            else:
                test.append(dat[1])
    
    return train, test


train, test = loadSplit()

def read_features(identity):
    face_feats_path = "D:\\saqlain\\faceFeaturesMat\\{}.mat".format(identity)
    features = loadmat(face_feats_path)
    return features['data']

uniqueness_dict ={}
train_feats=[]
for train_counter in train:
    #get train id
    train_id = train_counter.split('/')[0]
    # read features
    features = read_features(train_id)
    #append features
    index_to_save = uniqueness_dict.get(train_id, 0) + 1
    uniqueness_dict[train_id] = index_to_save
    train_feats.append(np.append(features[0][0], int(train_id.split('id1')[1])))

pd.DataFrame(train_feats).to_csv('face_train.csv', header=None, index=None)

test_feats= []
for test_counter in test:
    #get train id
    test_id = test_counter.split('/')[0]
    # read features
    features = read_features(test_id)
    #append features
    try:
        index_to_save = uniqueness_dict.get(test_id, 0) + 1
        uniqueness_dict[test_id] = index_to_save
        test_feats.append(np.append(features[0][0], int(test_id.split('id1')[1])))
    except Exception as excp:
        index_to_save=0
        uniqueness_dict[test_id] = index_to_save
        test_feats.append(np.append(features[0][0], int(test_id)))
        
pd.DataFrame(test_feats).to_csv('face_test.csv', header=None, index=None)


    
# voice_path = 'D:/voiceFeatsIdentification'
# image_path = 'D:/saqlain/faceFeaturesMat'

# face_feats_path = "D:\\saqlain\\faceFeaturesMat"
# list_face_feats = os.listdir(face_feats_path)

# facefeat = loadmat(face_feats_path + '\\' + list_face_feats[0])
# within each facefeat contains features for multiples files