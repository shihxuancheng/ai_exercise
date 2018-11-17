"""
Lab-Advertising

"""


import numpy as np
import pandas as pd
import math
from sklearn import linear_model

advertising = pd.read_csv('Advertising.csv')
reg = linear_model.LinearRegression()
X=advertising.iloc[:,1:4]
Y=advertising.iloc[:,4]
advertising['TV_radio']=advertising.TV*advertising.radio
reg.fit(X.values, Y.values)
print('W_o, W_tv, W_radio, W_newspaper: ', reg.intercept_, reg.coef_)
print('RSS: ', np.sum((reg.predict(X.values)-Y.values)**2))
print('R2: ', reg.score(X.values, Y.values))

temp=np.ones(advertising.shape[0])
temp1=temp[:,np.newaxis]
X_mat=np.append(temp1,X.values,axis=1)
W=np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y.values)
print('The optimal W is',W)

Sigma_W=np.linalg.inv(X_mat.T.dot(X_mat))*2.824424
print('Covariance: ', np.sqrt(Sigma_W[0,0]), np.sqrt(Sigma_W[1,1]), np.sqrt(Sigma_W[2,2]), np.sqrt(Sigma_W[3,3]))  
print('\n')

#=================================================================

X_new=advertising.iloc[:,[1,2,5]]
reg.fit(X_new.values, Y.values)
print('W_o, W_tv, W_radio, W_tvXradio: ', reg.intercept_, reg.coef_)
print('RSS: ', np.sum((reg.predict(X_new.values)-Y.values)**2))
print('R2: ', reg.score(X_new.values, Y.values))

temp=np.ones(advertising.shape[0])
temp1=temp[:,np.newaxis]
X_mat=np.append(temp1,X_new.values,axis=1)
W=np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(Y.values)
print('The optimal W is',W)
