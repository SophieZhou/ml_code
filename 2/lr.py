# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:31:53 2021

@author: Administrator
"""

import numpy as np

import matplotlib.pyplot as plt

# Size of the points dataset.
m = 20

# Points x-coordinate and dummy value (x0, x1).

X = np.arange(1, m+1).reshape(m, 1)


# Points y-coordinate
y = np.array([   3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
              11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(m, 1)


# The Learning Rate lr.
lr = 0.01

iter_all = 1000

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

def grad_w(w,b, X, y):
    '''Gradient of w.'''
    diff = np.dot(X, w) + b - y
    return (1./m) * np.dot(np.transpose(X), diff)



def grad_b(w,b, X, y):
    '''Gradient of b.'''  
    diff =  np.mean(np.dot(X, w) + b - y)
    print(diff)
    return diff



w=1
b=1
num = 0

while num<iter_all:
    w_old = w
    b_old = b
    w = w_old - lr*grad_w(w_old,b_old, X,y)
    b = b_old - lr*grad_b(w_old,b_old, X,y)
    num = num+1
        
        
print('optimal:', w,b)

yp=X*w+b
#print(yp)

plt.scatter(X,y,c='k')
plt.plot(X,yp,'b')
plt.savefig(r'D:\lixin-classes\2021-2022-1\机器学习与数据挖掘\zhoujy/gd_lr.png')
plt.show()






