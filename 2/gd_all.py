# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 17:01:39 2021

@author: Administrator
"""
# In[]
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.power(x, 2)

def d_f_1(x):
    return 2.0 * x

def d_f_2(f, x, delta=1e-4):
    return (f(x+delta) - f(x-delta)) / (2 * delta)


# plot the function
xs = np.arange(-10, 11)
plt.plot(xs, f(xs),c='k')


lr = 0.1
max_loop = 50

x_init = -10.0
x = x_init
lr = 0.1
x_s = [i for i in range(max_loop)]
y_s = [i for i in range(max_loop)]

for i in range(max_loop):
    x_s[i] = x
    y_s[i] = f(x)
    d_f_x = d_f_1(x)
    x = x - lr * d_f_x
    print(x)

print('initial x =', x_init)
print('mini f(x) when  x =', x)
print('mini f(x) =', f(x))

colors=[i*max_loop for i in range(max_loop)]
plt.scatter(x_s,y_s,c=colors,s=160,marker='+')


plt.savefig(r'gd.png')
plt.show()

# In[]
import numpy as np

import matplotlib.pyplot as plt

# Size of the points dataset.
m = 20

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))

# Points y-coordinate
y = np.array([   3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
              11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(m, 1)


# The Learning Rate lr.
lr = 0.01

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-6):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

optimal = gradient_descent(X, y, lr)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])

py=X1*optimal[1]+optimal[0]

plt.scatter(X1,y,c='k')
plt.plot(X1,py,'b')
plt.savefig(r'D:\lixin-classes\2021-2022-1\机器学习与数据挖掘\zhoujy/gdwb.png')
plt.show()
# In[]

import numpy as np

import matplotlib.pyplot as plt

# Size of the points dataset.
m = 20
X = np.arange(1, m+1).reshape(m, 1)


# Points y-coordinate
y = np.array([   3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
              11, 13, 13, 16, 17, 18, 17, 19, 21]).reshape(m, 1)
# The Learning Rate lr.
lr = 0.01
iter_all = 100

def error_func(w,b, X, y):
    '''Error function.'''
    diff = np.dot(X, w) + b - y
    return (1./m) * np.dot(np.transpose(diff), diff)

def grad_w(w,b, X, y):
    '''Gradient of w.'''
    diff = np.dot(X, w) + b - y
    return (1./m) * np.dot(np.transpose(X), diff)



def grad_b(w,b, X, y):
    '''Gradient of b.'''  
    diff =  np.mean(np.dot(X, w) + b - y)
 #   print(diff)
    return diff
w=1
b=1
num = 0
dis_step = 10
mse=[]
w_h = []
b_h=[]
while num<iter_all:
    w_old = w
    b_old = b
    w = w_old - lr*grad_w(w_old,b_old, X,y)
    b = b_old - lr*grad_b(w_old,b_old, X,y)
    num = num+1
    loss = error_func(w,b,X,y)
    mse.append(loss)
    w_h.append(w)
    b_h.append(b)
    if num % dis_step ==0:
        print('iter:{},loss:{},w:{},b:{}'.format(num,loss,w,b))
        
        
print('optimal:', w,b)

yp=X*w+b
#print(yp)


colors=[i*iter_all for i in range(iter_all)]
for i in range(iter_all):
    c = i
    plt.plot(X,X*w_h[i]+b_h[i],c='g')
    
plt.scatter(X,y,c='k')
plt.plot(X,yp,c='r')    
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.savefig(r'D:\lixin-classes\2021-2022-1\机器学习与数据挖掘\zhoujy/gd_lr.png')

plt.show()


plt.figure()

plt.plot(range(0,num),np.array(mse).reshape(num, 1),marker='+',c='b')

plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig(r'D:\lixin-classes\2021-2022-1\机器学习与数据挖掘\zhoujy/gd_loss.png')
plt.show()


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
#x, y = np.random.rand(10), np.random.rand(10)
#z = (np.random.rand(9000000)+np.linspace(0,1, 9000000)).reshape(3000, 3000)
plt.imshow(np.array(mse).reshape(num, 1), extent=(np.amin(w_h), np.amax(b_h), np.amin(w_h), np.amax(b_h)),
    cmap=cm.hot, norm=LogNorm())
plt.colorbar()
plt.savefig(r'D:\lixin-classes\2021-2022-1\机器学习与数据挖掘\zhoujy/gd_ht.png')
plt.show()




# In[]


x=np.array([1,2,3]).reshape(3,1)
x0=np.array([1,1,1]).reshape(3,1)

print(np.hstack([x,x0]))










