# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:11:59 2022

@author: Administrator
"""
# In[]
def fun_score(y_t,y_p):
    num = 0
    for i in range(len(y_p)):
        if y_t[i] == y_p[i]:
            num = num+1
        
    prec = num/len(y_p)
#    print(prec)
    return prec


# In[]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
 
iris = load_iris() 

plt.scatter(iris.data[:,2],iris.data[:,3],c=iris.target)
plt.show()
# In[]
X = iris.data  # 只包括样本的特征，150x4
Y = iris.target  # 样本的类型，[0, 1, 2]

X_train, X_test, y_train, y_test = train_test_split(X, Y)
print(len(X_train))

clf = AdaBoostClassifier(n_estimators=100) #迭代100次
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(fun_score(y_pred,y_test))



# In[]
scores = cross_val_score(clf, iris.data, iris.target) #分类器的精确度
scores.mean()        




# In[]
