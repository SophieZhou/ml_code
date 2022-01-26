# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:23:02 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 21:17:11 2022

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
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
import pandas as pd

data = load_iris()
pd.DataFrame(data=data.data, columns=data.feature_names)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

train_X = data.data  # 只包括样本的特征，150x4

train_Y = data.target  # 样本的类型，[0, 1, 2]
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y)
print(len(X_train))

# In[]
########## SVM training and comparison
# based on linear kernel as well as gaussian kernel
from sklearn import svm

for kernel in list(('linear', 'rbf')):
    # initial
    svc = svm.SVC(C=100, kernel=kernel) 
    # train
    svc.fit(X_train, y_train)
    # get support vectors
    sv = svc.support_vectors_
    
    y_pred = svc.predict(X_test)    
    
    precision = fun_score(y_test,y_pred)
    print(precision)
