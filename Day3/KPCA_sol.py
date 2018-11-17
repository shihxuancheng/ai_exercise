# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:10:24 2018

@author: Hsien-I Lin
"""
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



X = X.T
num_t = X.shape[1]
X_f = np.zeros((3,num_t), dtype='f')
for i in range(num_t):
    X_f[0,i] = X[0,i]**2
    X_f[1,i] = X[1,i]**2
    X_f[2,i] = np.sqrt(2)*X[0,i]*X[1,i]
    

U, s, V = np.linalg.svd(X_f, full_matrices=False)




Sigma = np.array([[s[0],0],[0,s[1]]])
Y_new = Sigma.dot(V[0:2,:])
Y_new = Y_new.T

plt.scatter(Y_new[reds, 0], Y_new[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(Y_new[blues, 0], Y_new[blues, 1], c="blue", s=20, edgecolor='k')

Kernel_mat = X_f.T.dot(X_f)    
eig_val_kernel, eig_vec_kernel = np.linalg.eig(Kernel_mat)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_kernel[i]), eig_vec_kernel[:,i]) for i in range(len(eig_val_kernel))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

v1 = eig_pairs[0][1]
v1 = v1[:,np.newaxis]

v2 = eig_pairs[1][1]
v2 = v2[:,np.newaxis]

V_k =  np.hstack((v1, v2))
Sigma_k = np.array([[np.sqrt(eig_pairs[0][0]),0],[0,np.sqrt(eig_pairs[1][0])]])
Y_new_k = Sigma_k.dot(V_k.T)
Y_new_k = Y_new_k.T

plt.figure()
plt.scatter(Y_new_k[reds, 0], Y_new_k[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(Y_new_k[blues, 0], Y_new_k[blues, 1], c="blue", s=20, edgecolor='k')


