# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:50:40 2022

@author: Hi
"""
import pandas as pd
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

#Grid Search CV
from sklearn.svm import SVC
#clf = SVC(kernel='poly')
clf = SVC()
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)

#---------------------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

param_grid = [{'kernel':['poly','rbf'],'gamma':[0.5,0.1,0.01],'C':[10,0.1,0.001,0.0001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)

clf = SVC(C= 15, gamma = 100)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)



