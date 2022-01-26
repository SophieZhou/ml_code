# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 14:45:00 2022

@author: Administrator
"""

# In[]

import numpy as np

import csv


label_dict = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

def loadDataset(filename, split_size):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        np.random.shuffle(dataset) # 第一个维度上打乱顺序
        num_line = len(dataset) # 106 /10   10,10,...10,16
        each_size = int((num_line+1) / split_size) #size of each split sets
        split_all = []
        each_split  = []
        count_num = 0 #count_num 统计每次遍历的当前个数                                   
        count_split = 0  #count_split 统计切分次数
        #print(len(dataset))
        
        for i in range(len(dataset)):
            each_split.append(dataset[i]) 
            count_num += 1
           # print(count_split)
            if  i < each_size*split_size and count_num == each_size:
                count_split = count_split + 1 
             #   print(count_split)
                array_ = np.array(each_split)            
                np.savetxt(outdir + "/split_" + str(count_split) + '.txt',\
                        array_,fmt="%s", delimiter='\t')  #输出每一份数据
                split_all.append(each_split) #将每一份数据加入到一个list中
                each_split = []
                count_num = 0
        else:
            if len(each_split)>0:
                array_ = np.array(each_split)
                np.savetxt(outdir + "/split_" + str(count_split) + '.txt',\
                        array_,fmt="%s", delimiter='\t')  
                split_all.append(each_split)
            
    return split_all
         

if __name__=='__main__':
    filename = r'iris.data'
    split_size = 10
    outdir = r'D:\data-science\ml\1//'
    split_all = loadDataset(filename, split_size)
    print(len(split_all))
    
    
    
    
    
    
    
    
    
# In[]
filename = r'iris.data'
label_dict = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

with open(filename, 'r') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    print(dataset)
    np.random.shuffle(dataset)
    print(dataset)
    for y in range(4):
        dataset[0][y] = float(label_dict[dataset[0][y]])
        print(dataset[0][y])
    
# In[]
dataset = [[1,2,3,4],[5,6,7,8],[10,20,30,40],[50,60,70,80]]
print(dataset)
print(np.random.shuffle(dataset))
print(dataset)
    
