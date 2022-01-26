# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:28:20 2021

@author: zhoujy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def sigmoid(x):
    f = 1.0 / (1+np.exp(-x))
    return f

"""
# define the net, one hidden layer with 4 neurons. 
#one input layer with dim=2, one output
#    layer with one neuron.
""" 
class Net():
    def __init__(self,input_num=2,hidden_num=4,output_num=1):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.w1 = np.random.rand(self.input_num, self.hidden_num)
        self.b1 = np.zeros((1, self.hidden_num))
        self.w2 = np.random.rand(self.hidden_num, self.output_num)
        self.b2 = np.zeros((1, self.output_num))

    def forward(self, X):
        """
        Forward process of this simple network.
        Input:
            X: np.array with shape [N, 2]
        """
        self.X = X
        self.z1 = X.dot(self.w1) + self.b1  # sum with weights,input of the hidden layer
        self.h = sigmoid(self.z1)  # output of the hidden layer
        self.z2 = self.h.dot(self.w2) + self.b2  # input of the output layer
        self.y = sigmoid(self.z2) #output of the output layer

    def grad(self, Y):
        """
        Compute gradient of parameters for training data (X, Y). 
        X is saved in self.X.
        """
        grad_z2 = self.y - Y
        self.grad_w2 = self.h.T.dot(grad_z2)
        self.grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
        
        grad_h = grad_z2.dot(self.w2.T)
        grad_z1 = grad_h * self.h * (1-self.h)
        
        self.grad_w1 = self.X.T.dot(grad_z1)
        self.grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
        
    def update(self, lr=0.1):
        """
        Update parameters with gradients.
        """
        self.w1 -= lr*self.grad_w1
        self.b1 -= lr*self.grad_b1
        self.w2 -= lr*self.grad_w2
        self.b2 -= lr*self.grad_b2

    def bp(self, Y):
        """
        BP algorithm on data (X, Y)
        Input:
            Y: np.array with shape [N, 1]
        """
        self.grad(Y)
        self.update()

    def loss(self, X, Y):
        """
        Compute loss on X with current model.
        Input:
            X: np.array with shape [N, 2]
            Y: np.array with shape [N, 1]
        Return:
            cost: float
        """
        self.forward(X)
      #  print('predict score',self.y)
        cost = np.sum(-np.log(self.y) * Y)
        return cost


if __name__=="__main__":

    #read data from csv data file
    train_X = []
    train_Y = []
    data = pd.read_csv("./watermelon_3a.csv")
    X1 = data.values[:, 1]
    X2 = data.values[:, 2]
    y = data.values[:, 3]
    print(y)
    N = len(X1)
    for i in range(N):
        train_X.append(np.array([X1[i], X2[i]]))        
        train_Y.append(y[i])

    print(train_Y)
    #check grads
    net = Net(input_num=2,hidden_num=4,output_num=1)

    #training
    iterations = 5000
    train_Xs = np.vstack(train_X)
    train_Ys = np.vstack(train_Y)
    #training network with standard BP
    standard_net = Net()
    train_losses = []
    for it in range(iterations):
        for i in range(N):
            standard_net.forward(train_X[i].reshape(1, 2))
            standard_net.bp(train_Y[i].reshape(1))
        
        loss = standard_net.loss(train_Xs, train_Ys)        
        train_losses.append(loss)
    line1, = plt.plot(range(iterations), train_losses, "r-")
  
    
    
    #training network with accumulated BP
    accumulated_net = Net()
    train_losses = []
    for iter in range(iterations):
        accumulated_net.forward(train_Xs)
        accumulated_net.bp(train_Ys)
        loss = accumulated_net.loss(train_Xs, train_Ys)
        train_losses.append(loss)
    line2, = plt.plot(range(iterations), train_losses, "b-")
    plt.legend([line1, line2], ["BP", "Accumulated BP"])
    plt.show()