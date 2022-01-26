# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:56:00 2022

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
# reshape targets to get column vector with shape (n_samples, 1)
y_true = y_true[:, np.newaxis]
# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_true)

print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')

# In[]
class SoftmaxRegressor:
#
#    def __init__(self):
#        pass

    def train(self, X, y_true, n_classes, n_iters=10, learning_rate=0.1):
        """
        Trains a multinomial logistic regression
        """
        self.n_samples, n_features = X.shape
        self.n_classes = n_classes
        
        self.weights = np.random.rand(self.n_classes, n_features)
        self.bias = np.zeros((1, self.n_classes))
        all_losses = []
        
        for i in range(n_iters):
            scores = self.compute_scores(X) # X*W^T+b
            probs = self.softmax(scores) # exp()/sum(exp)
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
            y_one_hot = self.one_hot(y_true) # [0,1,0] [1,0,0],[0,0,1]

            loss = self.cross_entropy(y_one_hot, probs)
            all_losses.append(loss)

            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)

            self.weights = self.weights - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db

            if i % 100 == 0:
                print(f'Iteration number: {i}, loss: {np.round(loss, 4)}')

        return self.weights, self.bias, all_losses

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            numpy array of shape (n_samples, 1) with predicted classes
        """
        scores = self.compute_scores(X)
        probs = self.softmax(scores)  # [0.1,0.01,0.89]
        return np.argmax(probs, axis=1)[:, np.newaxis]

    def softmax(self, scores):
        """
        Tranforms matrix of predicted scores to matrix of probabilities
        
        Args:
            scores: numpy array of shape (n_samples, n_classes)
            with unnormalized scores
        Returns:
            softmax: numpy array of shape (n_samples, n_classes)
            with probabilities
        """
        exp = np.exp(scores)
        sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
        softmax = exp / sum_exp
        
        return softmax

    def compute_scores(self, X):
        """
        Computes class-scores for samples in X
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            scores: numpy array of shape (n_samples, n_classes)
        """
        return np.dot(X, self.weights.T) + self.bias

    def cross_entropy(self, y_true, probs):
        loss = - (1 / self.n_samples) * np.sum(y_true * np.log(probs))
        return loss

    def one_hot(self, y):
        """
        Tranforms vector y of labels to one-hot encoded matrix
        """
        one_hot = np.zeros((self.n_samples, self.n_classes))
        one_hot[np.arange(self.n_samples), y.T] = 1
      #  print(one_hot)
        return one_hot

# In[]
regressor = SoftmaxRegressor()
w_trained, b_trained, loss = regressor.train(X_train, y_train, learning_rate=0.1, n_iters=800, n_classes=4)

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(800), loss)
plt.title("Development of loss during training")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.show()



# In[]
n_test_samples, _ = X_test.shape
y_predict = regressor.predict(X_test)
print(f"Classification accuracy on test set: {(np.sum(y_predict[0] == y_test)/n_test_samples) * 100}%")        

# In[]

iris = datasets.load_iris()
X = iris.data[:, 2:]  # we only take the last two features.
y = iris.target

from sklearn.model_selection import train_test_split


X_iristrain, X_iristest, y_iristrain, y_iristest = train_test_split(X, y, test_size=0.25, random_state=2)

#print(y_iristest)



# In[]
regressor = SoftmaxRegressor()
w_trained, b_trained, loss = regressor.train(X_iristrain, y_iristrain, learning_rate=0.1, n_iters=10000, n_classes=3)

# In[]
n_test_samples, _ = X_iristest.shape
print(n_test_samples)
y_predict = regressor.predict(X_iristest)
scores = regressor.compute_scores(X_iristest)
y_score = regressor.softmax(scores)
print((y_score))
print((y_predict))
#print(np.sum(y_predict[0] == y_iristest))
#print(f"Classification accuracy on test set: {(np.sum(y_predict[0] == y_iristest)/n_test_samples) * 100}%")        


# In[]
"""
scikit learn

"""

from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(C=10)
classifier = LogisticRegression(multi_class="multinomial", solver="newton-cg", C=10)
classifier.fit(X_iristrain, y_iristrain)

test_results = classifier.predict(X_iristest)

n_test_samples, _ = X_iristest.shape
print(n_test_samples)
#print((y_predict))
#print((y_iristest))
print(np.sum(test_results[0] == y_iristest))
print(f"Classification accuracy on test set: {(np.sum(test_results[0] == y_iristest)/n_test_samples) * 100}%")        
