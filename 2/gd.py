# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.power(x, 2)

def d_f_1(x):
    return 2.0 * x


xs = np.arange(-10, 11)
plt.plot(xs, f(xs),c='k')


lr = 0.1
max_loop = 50
x_init = 10.0
x = x_init
x_s = [i for i in range(max_loop)]
y_s = [i for i in range(max_loop)]

num=0
while num<max_loop:
    x_s[num] = x
    y_s[num] = f(x)
    # gradient
    grd = d_f_1(x)
    # update 
    x = x - lr * grd
    if abs(f(x) - y_s[num])<=0.000001 or abs(grd)<=0.00001:
        break
    num = num+1
    print(x)

print('initial x =', x_init)
print('mini f(x) when  x =', x)
print('mini f(x) =', f(x))
print(num)
colors=[i*num for i in range(num)]
plt.scatter(x_s[:num],y_s[:num],c=colors,s=160,marker='+')

plt.savefig(r'gd.png')
plt.show()


