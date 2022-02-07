# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:53:04 2021

@author: guest
"""

# IMPORTING PACKAGES

import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
from termcolor import colored as cl # text customization
import itertools # advanced tools

from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split # data split
from sklearn.tree import DecisionTreeClassifier # Decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.linear_model import LogisticRegression # Logistic regression algorithm
from sklearn.svm import SVC # SVM algorithm
from sklearn.ensemble import RandomForestClassifier # Random forest tree algorithm
from xgboost import XGBClassifier # XGBoost algorithm

from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import f1_score # evaluation metric

#####
df=pd.read_csv('creditcard.csv')
 
x=df.drop('Class',axis=1).values
y=df['Class'].values
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# 1. knn
knn_model=KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(X_train, y_train)
knn_predicted=knn_model.predict(X_test)
#2. SVM
svm_model=SVC()
svm_model.fit(X_train,y_train)
svm_predicted=svm_model.predict(X_test)
# 1. Accuracy score
print(cl('ACCURACY SCORE', attrs = ['bold']))

print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the KNN model is {}'.format(accuracy_score(y_test, knn_predicted)), attrs = ['bold'], color = 'green'))

print(cl('------------------------------------------------------------------------', attrs = ['bold']))
print(cl('Accuracy score of the SVM model is {}'.format(accuracy_score(y_test, svm_predicted)), attrs = ['bold']))
print(cl('------------------------------------------------------------------------', attrs = ['bold']))
