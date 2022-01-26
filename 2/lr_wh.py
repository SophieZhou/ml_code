# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 21:10:38 2022

@author: Administrator
"""
import pandas as pd
from io import StringIO
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# 体重身高数据(csv文件)
csv_data = 'height,weight\n58,115\n59,117\n60,120\n61,123\n62,126\n63,129\n \
64,132\n65,135\n66,139\n67,142\n68,146\n69,150\n70,154\n71,159\n72,164\n'

# 读入dataframe
df = pd.read_csv(StringIO(csv_data))
#print(df)
dftest=df
# 建立线性回归模型
regr = linear_model.LinearRegression()

# 线性回归拟合
regr.fit(df['height'].values.reshape(-1, 1), df['weight']) 


# 画图
# 1.真实的点
plt.scatter(df['height'], df['weight'], color='blue')

# 2.linear regression拟合的直线
plt.plot(df['height'], regr.predict(df['height'].values.reshape(-1,1)), color='green', linewidth=4)
x1 = np.array([75])
x2 = np.array([67])
###预测的值
print(x1,regr.predict(x1.reshape(-1,1)))
print(x2,regr.predict(x2.reshape(-1,1)))
plt.savefig('lr.jpg')
plt.show()
