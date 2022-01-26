# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:14:09 2022

@author: Administrator
"""


# In[]

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 

lris_df = datasets.load_iris()
x_axis = lris_df.data[:,0]
y_axis = lris_df.data[:,2] 
 

model = KMeans(n_clusters=3) 
#训练模型
model.fit(lris_df.data)
prddicted_label= model.predict([[6.3, 3.3, 6, 2.5]])
all_predictions = model.predict(lris_df.data) 
plt.scatter(x_axis, y_axis, c=all_predictions)
plt.show()