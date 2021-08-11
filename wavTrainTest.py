# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:40:44 2021

@author: Muhammad Saad Saeed (18F-MS-CP-01)


"""

from glob import glob
import os
from scipy.io import loadmat
import numpy as np
import pandas as pd

def path_to_disk():
    voice = []
    
    for ids in glob('E:/UETGen/VoxCeleb/wav/*'):
        print(ids)
        for links in glob(ids+'/*'):
            for wav in glob(links+'/*'):
                voice.append(wav)
                    
    return voice
    

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
voicePath = path_to_disk()

# In[1]:
path = "D:\\voiceFeatsIdentification"

def loadMat(voicePath):
    feat = {}
    counter = 0
    for mats in glob('D:/voiceFeatsIdentification/*'):
        print(mats)
        tmpMat = loadmat(mats)
        tmpMat = tmpMat[list(tmpMat.keys())[-1]]
        for tmpFeat in tmpMat:
            tmpFeat = tmpFeat[0]
            pathTmp = voicePath[counter][-29:].replace('\\','/')
            feat[pathTmp] = tmpFeat
            counter+=1
    return feat
            
feats = loadMat(voicePath)


# In[2]:
    
trainFeats = []
testFeats = []

for pathTmp in test:
    testFeats.append(np.append(feats[pathTmp], int(pathTmp.split('/')[0].split('id1')[1])))
    
for pathTmp in train:
    trainFeats.append(np.append(feats[pathTmp], int(pathTmp.split('/')[0].split('id1')[1])))
    
    
    
# In[3]:
pd.DataFrame(testFeats).to_csv('wav_test.csv', header=None, index=None)
pd.DataFrame(trainFeats).to_csv('wav_train.csv', header=None, index=None)



