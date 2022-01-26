# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:33:37 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 19:44:38 2022

@author: Administrator
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
x, y = iris.data, iris.target
kf =KFold(n_splits=10, shuffle=True)

result=[]
dit = {}
for k,(train,test) in enumerate(kf.split(x,y)):      
    test_score=[]
    x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
    print("train_split_rate:",len(x_train)/len(x))
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    print(knn.score(x_test, y_test))
        
    print("第%s轮验证："%(k+1))
    print("本轮训练集得分：%.2f%%"%(knn.score(x_train,y_train)*100))
    print("本轮测试集得分：%.2f%%"%(knn.score(x_test,y_test)*100))
    test_score.append(knn.score(x_test,y_test))
result.append(np.mean(test_score))   
print("mean_score:",np.mean(test_score))
print('bets mean recall score:',max(result))


