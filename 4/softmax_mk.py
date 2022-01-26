# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:17:01 2022

@author: Administrator
"""

# In[]
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
np.random.seed(13)
X, y_true = make_blobs(centers=4, n_samples = 5000)

# In[]
fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y_true)
plt.title("Dataset")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
# In[]

from softmax_iris import one_hot
# reshape targets to get column vector with shape (n_samples, 1)
y_true = y_true[:, np.newaxis]
# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_true)

#print(f'Shape X_train: {X_train.shape}')
#print(f'Shape y_train: {y_train.shape}')
#print(f'Shape X_test: {X_test.shape}')
#print(f'Shape y_test: {y_test}')  

y_train = one_hot(y_train,4)
y_test = one_hot(y_test,4)

print(y_test)
# In[]
from softmax_iris import Net 
input_num=2
hidden_num=8
output_num=4
N = len(X_train)


lr = 0.1
    #training
iterations = 1000
    #training network with standard BP
standard_net = Net(input_num=2,hidden_num=6,output_num=4)
train_losses = []
for iter in range(iterations): # 1000
    for i in range(N): # 5000
        standard_net.forward(X_train[i].reshape(1, input_num))
        standard_net.bp(y_train[i],lr)
    loss = standard_net.loss(X_train, y_train)
    print(loss)
    train_losses.append(loss)
line1, = plt.plot(range(iterations), train_losses, "r-")

# In[]
standard_net.forward(X_test)
yy = np.argmax(y_test, axis=1)
yp = np.argmax(standard_net.y, axis=1)

num = 0
for i in range(len(yy)):
    if yy[i] == yp[i]:
        num = num+1
        
prec = num/len(yy)
print(prec)

# In[]
standard_net.precision(X_test,y_test)