# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 07:21:25 2018

kaggle_test.py

"""

import urllib.request
import os
import pandas as pd
from sklearn import preprocessing
import numpy as np

#url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
#filepath = "data/titanic3.xls"
#if not os.path.isfile(filepath):
#    result = urllib.request.urlretrieve(url,filepath)
#    print("download",result)
all_df = pd.read_excel("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls")
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
all_df = all_df[cols]

train_df = all_df.sample(frac=0.8, random_state=99)
test_df = all_df.loc[~all_df.index.isin(train_df.index),:]


#=====Test
Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0, 'S'])
Rose = pd.Series([1, 'Jack', 1, 'female', 20, 1, 0, 100.0, 'S'])
JR_df = pd.DataFrame([list(Jack), list(Rose)], columns=['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])
all_df = pd.concat([all_df, JR_df])
#=====Test

def PreprocessData(all_df):
    df = all_df.drop('name', axis=1)
    df.isnull().sum()
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)

    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    Onehot_df = pd.get_dummies(data=df, columns=["embarked"])
    dfarray = Onehot_df.values
    dfarray.shape
    Labels = dfarray[:,0]
    Features = dfarray[:,1:]
    #----------
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaledFeatures = minmax_scale.fit_transform(Features)
    #(np.where(np.isnan(Features)))[0].shape[0]
    #Onehot_df.loc[1225,:]
    return scaledFeatures, Labels


train_feature, train_label= PreprocessData(train_df)
test_feature, test_label = PreprocessData(test_df)
JR_feature, JR_label= PreprocessData(JR_df)

from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

def show_train_history(train_history, train, validation, title):
    plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show
    

model = Sequential()
model.add(Dense(units=40, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_result = model.fit(x=train_feature, y=train_label, validation_split=0.1, epochs=30, batch_size=30, verbose=2)
show_train_history(train_result, 'acc', 'val_acc', 'Trained accuracy')
show_train_history(train_result, 'loss', 'val_loss', 'Trained loss')
scores = model.evaluate(x=test_feature, y=test_label)
scores[1]

#==== Prediction
all_feature, all_label= PreprocessData(all_df)
all_probability = model.predict(all_feature)
all_probability[-2:]
pd = all_df
pd.insert(len(all_df.columns), 'probability', all_probability)
pd[-2:]