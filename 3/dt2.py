# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 07:53:19 2022

@author: Administrator
"""


import pandas as pd
from math import log


# 
def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    return datasets, labels



# entropy  -sum(pklog(pk,2))
def entropy(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label]+=1
#    ent = -sum([( p/data_length) * log(p/data_length,2)
#                for p in label_count.values()])
    ent_sum = 0
    for p in label_count.values():
        pk = p/data_length
        entmp = pk * log(pk,2)
        ent_sum = ent_sum + entmp
    ent = -ent_sum     
    return ent

# sum((dv/d)* ent(dv))
def condition_entropy(datasets,axis = 0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i]) # Dv
   # condi_ent = sum([ (len(p) / data_length)*entropy(p) for p in feature_sets.values()])
    condi_ent = 0
    for p in feature_sets.values(): # p 
        entp = entropy(p)  # ent(Dv)
        conent = entp * len(p)/data_length   # Dv/D
        condi_ent = condi_ent + conent
    
    return condi_ent

# 信息增益
def info_gain(ent,condi_entropy):
    return ent - condi_entropy

def info_gain_train(data_sets):
    count = len(datasets[0]) - 1  # feature dim 
    ent = entropy(datasets)
    
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent,condition_entropy(datasets,axis=c))
        best_feature.append((c,c_info_gain))
        print("特征（{}）的信息增益为： {:.3f}".format(labels[c],c_info_gain))
    best = max(best_feature, key=lambda x:x[-1])
    print( '特征({})的信息增益最大，选择为根节点特征'.format(labels[best[0]]))

if __name__=='__main__':
    datasets, labels = create_data()
    train_data = pd.DataFrame(datasets, columns=labels)
    print(len(datasets[0]))
    info_gain_train(datasets)