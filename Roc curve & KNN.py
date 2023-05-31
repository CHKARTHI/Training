# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 14:22:39 2022

@author: DELL
"""

import pandas as pd
import numpy as np
X=pd.read_csv("breast_cancer.csv")
X
from sklearn.linear_model import LogisticRegression
X.corr()
#Step 2: Boxplotm visulization and cleaning
#Bargraph
#histogram
# Step 3: split in to x and y variables
#Data transforamtion
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(X["Class"])
y
list(y)
list(X)
x=X.iloc[:,1:10]
x

#4 model fitting
lgr=LogisticRegression()
#lgr.coef_
lgr.fit(x,y)
y_pred=lgr.predict(x)
y
y_pred


#5 error

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y,y_pred)
cm

ac=accuracy_score(y,y_pred)
ac.round(3)
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score


rc=recall_score(y,y_pred)
rc
pr=precision_score(y,y_pred)
f1=f1_score(y,y_pred)

cm
TN=cm[0,0]
TN
FP=cm[0,1]
Specificty=TN/(TN+FP)
Specificty

from sklearn.metrics import roc_auc_score, roc_curve
lgr.predict_proba(x).shape
y_pred_prob=lgr.predict_proba(x)[:,1]
fpr,tpr,_=roc_curve(y,y_pred_prob)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
#plt.Legend(Loc=4)
plt.ylable('tpr')
plt.xlabor('fpr')

Rocscore=roc_auc_score(y,y_pred_prob)
Rocscore

#testing..........................................

X=pd.read_csv("breast_cancer.csv")
X
from sklearn.linear_model import LogisticRegression
X.corr()
#Step 2: Boxplotm visulization and cleaning
#Bargraph
#histogram
# Step 3: split in to x and y variables
#Data transforamtion
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(X["Class"])
y
list(y)
list(X)
x=X.iloc[:,1:10]
x

#4-data partition..
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)
# print(x_train.shape)
# print(x_test.shape)

# print(x_train.shape)
# print(x_test.shape)

#5-model fitting..
from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression()
lgr.fit(x_train,y_train)
y_pred_train = lgr.predict(x_train)
y_pred_test = lgr.predict(x_test)


#6-accuracy score..

from sklearn.metrics import accuracy_score
trainacscore=accuracy_score(y_train,y_pred_train)
testacscore=accuracy_score(y_test,y_pred_test)
trainacscore
testacscore

...............................using KNN classification

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred_train = knn.predict(x_train)
y_pred_test = knn.predict(x_test)


#6-accuracy score..

from sklearn.metrics import accuracy_score
trainacscore=accuracy_score(y_train,y_pred_train)
testacscore=accuracy_score(y_test,y_pred_test)
trainacscore
testacscore

#random state x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, randdomstate=100)










































































