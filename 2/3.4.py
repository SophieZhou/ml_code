# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 19:44:38 2022

@author: Administrator
"""

# In[]

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score


data_path = r'./Transfusion.txt'

data = np.loadtxt(data_path, delimiter=',').astype(int)

X = data[:, :4]
y = data[:, 4]

m, n = X.shape


X = (X - X.mean(0)) / X.std(0)

# shuffle
index = np.arange(m)
np.random.shuffle(index)

X = X[index]
y = y[index]

# k-10 cross validation
lr = linear_model.LogisticRegression(C=2)

score = cross_val_score(lr, X, y, cv=5)
print(score)

print('k-cross validation results',score.mean())
# In[]
print(score)
# In[]
# LOO
loo = LeaveOneOut()
accuracy = 0
for train, test in loo.split(X, y):
    lr_ = linear_model.LogisticRegression(C=2)
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]
    lr_.fit(X_train, y_train)

    accuracy += lr_.score(X_test, y_test)

print('loo results:',accuracy / m)


print('k-cross validation results',score.mean())
# In[]
print('loo results:',accuracy / m)


print('k-cross validation results',score.mean())

# In[]

num_split = int(m / 10)
score_my = []
for i in range(10):
    lr_ = linear_model.LogisticRegression(C=2)
    test_index = range(i * num_split, (i + 1) * num_split)
    X_test_ = X[test_index]
    y_test_ = y[test_index]

    X_train_ = np.delete(X, test_index, axis=0)
    y_train_ = np.delete(y, test_index, axis=0)

    lr_.fit(X_train_, y_train_)

    score_my.append(lr_.score(X_test_, y_test_))

print(np.mean(score_my))

# LOO
score_my_loo = []
for i in range(m):
    lr_ = linear_model.LogisticRegression(C=2)
    X_test_ = X[i, :]
    y_test_ = y[i]

    X_train_ = np.delete(X, i, axis=0)
    y_train_ = np.delete(y, i, axis=0)

    lr_.fit(X_train_, y_train_)

    score_my_loo.append(int(lr_.predict(X_test_.reshape(1, -1)) == y_test_))

print(np.mean(score_my_loo))

