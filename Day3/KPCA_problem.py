# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:10:24 2018

@author: Hsien-I Lin
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)

reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[reds, 0]**2, X[reds, 1]**2, np.sqrt(2)*X[reds, 0]*X[reds, 1], color='r')
ax.scatter(X[blues, 0]**2, X[blues, 1]**2, np.sqrt(2)*X[blues, 0]*X[blues, 1], color='b')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()



