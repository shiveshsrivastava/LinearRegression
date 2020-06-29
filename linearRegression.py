# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:20:20 2020

@author: Admin
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

learnRate = 0.1

#data = pd.read_csv("housing.csv", usecols = ['RM', 'LSTAT'])

#Y = pd.read_csv("housing.csv", usecols = ['MEDV'])
#Y = np.array(Y)
#Y = np.reshape(Y, 489)

#data= pd.DataFrame(np.c_[data['LSTAT'], data['RM']], columns = ['LSTAT','RM'])

#dataArr = np.array(data)

#dataArr = np.insert(dataArr, 0, 1, axis=1)



x = np.random.rand(100, 1)
dataArr = np.insert(x, 0, 1, axis=1)
Y = 2 + 3 * x + np.random.rand(100, 1)
Y = np.reshape(Y, 100)
theta = np.array([1,1])
print(dataArr)
print(theta.shape)

print(x[10], Y[10])

m = len(Y)

for i in range(10000): #try changing the epochs to 100
    h = dataArr @ theta
    
    print(h.shape)
    
    sub = h-Y
    print(h.shape, Y.shape)
    
    J = (1/(2*m))*np.sum(sub**2)
    print(J)
    #dataArr = np.concatenate(np.ones(489), dataArr)
    
    #print((h-Y).shape)
    
    delta = (1/m) * (dataArr.T @ sub)
    
    #print(delta.shape)
    
    theta = theta - (learnRate * delta)
    
    error = np.sum(h-Y)
    print("ERROR: ", error)
    
#print(dataArr)    
plt.scatter(x,Y)
plt.plot(dataArr[:, 1:2], h)
plt.show()
    
print(theta[0] + (theta[1]*0.92348777))

"""  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

zline = h
xline = dataArr[:, 0:1]
yline = dataArr[:, 1:2]
print(xline)
ax.scatter(dataArr[:, 0:1], dataArr[:, 1:2], Y)
ax.plot(xline, yline, zline, zdir='z')
plt.show()
"""
