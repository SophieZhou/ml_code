# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 09:38:02 2022

@author: Administrator
"""
# In[]
import numpy as np
import csv
import random

"""
split train and test
"""

def loadDataset(filename, split, trainingSet = [], testSet = []):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile) 
        print(lines)
        dataset = list(lines)        
        for x in range(len(dataset)-1):
#            for y in range(4):
#                dataset[x][y] = float(dataset[x][y])
            
            if random.random() < split:  #将数据集随机划分
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                
if __name__=='__main__':
    trainingSet = []
    testSet = []
    split = 0.75 # train:test=2:1
    filename = r'D:\data-science\ml\1//iris.data'
    
    loadDataset(filename, split, trainingSet, testSet)
    print('Trainset: ' + repr(len(trainingSet)))
    print('Testset: ' + repr(len(testSet)))
    
    
    