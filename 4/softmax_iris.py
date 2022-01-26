# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 14:34:57 2022

@author: Administrator
"""


# In[]]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    f = 1.0 / (1+np.exp(-x))
    return f

def softmax(x):
    #array
    if len(x.shape)==2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        f = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    #vector
    else:
        x = x - np.max(x)
        exp_x = np.exp(x)
        f = exp_x / np.sum(exp_x)
    return f
# 4,6,3
class Net():
    def __init__(self,input_num=2,hidden_num=4,output_num=1):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.W1 = np.random.rand(self.input_num, self.hidden_num)
        self.b1 = np.zeros((1, self.hidden_num))
        self.W2 = np.random.rand(self.hidden_num, self.output_num)
        self.b2 = np.zeros((1, self.output_num))

    def forward(self, X):
        """
        Forward process of this simple network.
        Input:
            X: np.array with shape [N, 2]
        """
        self.X = X
        self.z1 = X.dot(self.W1) + self.b1
        self.h = sigmoid(self.z1)
        self.z2 = self.h.dot(self.W2) + self.b2
        self.y = softmax(self.z2) # softmax

    def grad(self, Y):
        """
        Compute gradient of parameters for training data (X, Y). X is saved in self.X.
        """
        # gradient of error
        grad_z2 = self.y - Y
        self.grad_W2 = self.h.T.dot(grad_z2)        
        self.grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
        #gradient of 
        grad_h = grad_z2.dot(self.W2.T)
        grad_z1 = grad_h * self.h * (1-self.h)
        self.grad_W1 = self.X.T.dot(grad_z1)
        self.grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
        #used for grad_check()
        self.grads = [self.grad_W1, self.grad_W2, self.grad_b1, self.grad_b2]

    def update(self, lr=0.1):
        """
        Update parameters with gradients.
        """
        self.W1 -= lr*self.grad_W1
        self.b1 -= lr*self.grad_b1
        self.W2 -= lr*self.grad_W2
        self.b2 -= lr*self.grad_b2

    def bp(self, Y,lr):
        """
        BP algorithm on data (X, Y)
        Input:
            Y: np.array with shape [N, 2]
        """
        self.grad(Y)
        self.update(lr)

    def loss(self, X, Y):
        """
        Compute loss on X with current model.
        Input:
            X: np.array with shape [N, 2]
            Y: np.array with shape [N, 2]
        Return:
            cost: float
        """
        self.forward(X)
        cost = np.sum(-np.log(self.y) * Y)
        return cost
    def precision(self,X,Y):
        self.forward(X)
        indx = np.argmax(self.y, axis=1)
        yy = np.argmax(Y, axis=1)
        prec = 0
        num = 0
        for i in range(len(Y)):
            if yy[i] == indx[i]:
                num = num+1
        prec = num/len(Y)
        return prec

# In[]
        
def one_hot(y,num_class):
    one_hot = np.zeros((y.shape[0], num_class))
    for i in range(len(y)):
        one_hot[i][y[i]]=1
    return one_hot
    
# In[]
from sklearn.datasets import load_iris
data = load_iris()
pd.DataFrame(data=data.data, columns=data.feature_names)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

train_X = data.data  # 只包括样本的特征，150x4
print(train_X)
train_Y = data.target  # 样本的类型，[0, 1, 2]

print(train_Y)
# Y=[1,0,0],[0,1,0],[0,0,1]
train_Y=one_hot(train_Y,3)

N = len(train_X)
input_num = train_X.shape[1]

# In[]
if __name__=="__main__":
    #net

    lr = 0.1
    #training
    iterations = 1000
    #training network with standard BP
    standard_net = Net(input_num=4,hidden_num=6,output_num=3)
    train_losses = []
    for iter in range(iterations):
        for i in range(N):                      
            standard_net.forward(train_X[i].reshape(1, input_num))
            standard_net.bp(train_Y[i],lr)
        loss = standard_net.loss(train_X, train_Y)
      #  print(loss)
        train_losses.append(loss)
    line1, = plt.plot(range(iterations), train_losses, "r-")
 
    #training network with accumulated BP    
    train_Xs = np.vstack(train_X)
    train_Ys = np.vstack(train_Y)
    accumulated_net = Net(input_num=4,hidden_num=6,output_num=3)
    train_losses = []
    for iter in range(iterations):
        accumulated_net.forward(train_Xs)
        accumulated_net.bp(train_Ys,lr)
        loss = accumulated_net.loss(train_Xs, train_Ys)
        train_losses.append(loss)
    line2, = plt.plot(range(iterations), train_losses, "b-")
    plt.legend([line1, line2], ["BP", "Accumulated BP"])
    plt.show()