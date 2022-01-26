# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 21:04:25 2022

@author: Administrator
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
data=pd.DataFrame({'square_feet':[150,200,250,300,350,400,600],
                   'price':[5450,6850,8750,9650,10450,13450,16450]})
#创建一组7行2列的数据，square_feet为房屋面积，price为对应价格
#这里是将数据转化为一个1维矩阵
data_train=np.array(data['square_feet']).reshape(data['square_feet'].shape[0],1)
data_test=data['price']

#help(LinearRegression)

regr=LinearRegression() #创建线性回归模型，参数默认
regr.fit(data_train,data_test)
print(regr.coef_)
a=regr.predict(np.array(268.5).reshape(-1, 1))
print(268.5,a)#查看预测结果
#评分函数，将返回一个小于1的得分
print(regr.score(data_train,data_test))


plt.scatter(data['square_feet'],data['price']) 
plt.plot(data['square_feet'],regr.predict(np.array(data['square_feet']).reshape(data['square_feet'].shape[0],1)),color='red') 
