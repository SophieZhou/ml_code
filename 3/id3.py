# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 07:32:37 2022

@author: Administrator
"""


# using pandas dataframe for .csv read which contains chinese char.
import pandas as pd

data_file_encode = "gb18030"  # the watermelon_3.csv is file codec type
with open("./watermelon_3.csv", mode='r', encoding=data_file_encode) as data_file:
    df = pd.read_csv(data_file)



'''
implementation of ID3

rely on decision_tree.py
'''
import decision_tree4_3 as decision_tree

root = decision_tree.TreeGenerate(df)

# df = df.drop(['密度','含糖率'], 1)
# df = df.drop(['色泽','根蒂','敲声','纹理','脐部','触感'], 1)

accuracy_scores = []

'''
from random import sample
for i in range(10):
    train = sample(range(len(df.index)), int(1*len(df.index)/2))

    df_train = df.iloc[train]
    df_test = df.drop(train)
    # generate the tree
    root = decision_tree.TreeGenerate(df_train)
    # test the accuracy
    pred_true = 0
    for i in df_test.index:
        label = decision_tree.Predict(root, df[df.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1

    accuracy = pred_true / len(df_test.index)
    accuracy_scores.append(accuracy)
'''

# k-folds cross prediction

n = len(df.index)
k = 5
for i in range(k):
    m = int(n / k)
    test = []
    for j in range(i * m, i * m + m):
        test.append(j)

    df_train = df.drop(test)
    df_test = df.iloc[test]
    root = decision_tree.TreeGenerate(df_train)  # generate the tree

    # test the accuracy
    pred_true = 0
    for i in df_test.index:
        label = decision_tree.Predict(root, df[df.index == i])
        if label == df_test[df_test.columns[-1]][i]:
            pred_true += 1

    accuracy = pred_true / len(df_test.index)
    accuracy_scores.append(accuracy)

# print the prediction accuracy result
accuracy_sum = 0
print("accuracy: ", end="")
for i in range(k):
    print("%.3f  " % accuracy_scores[i], end="")
    accuracy_sum += accuracy_scores[i]
print("\naverage accuracy: %.3f" % (accuracy_sum / k))

# dicision tree visualization using pydotplus.graphviz
root = decision_tree.TreeGenerate(df)

#decision_tree.DrawPNG(root, "decision_tree_ID3.png")