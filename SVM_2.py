# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:10:54 2022

@author: Hi
"""

import pandas as pd
df = pd.read_csv("createdata.csv")
df.shape
df.head()

df.corr()

X = df.iloc[:,0:2]
y = df.iloc[:,3]


# Splitting Train and Test
from sklearn.model_selection._split import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Loading SVC 
# Training a classifier - kernel='rbf'
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0)
# clf = SVC(kernel='linear',C=2.0)
# clf = SVC(kernel='poly',degree=3)
# clf = SVC(kernel='rbf', gamma=4)

clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

# import the metrics class
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred_test)
print(cm)

print("Training Accuracy :",metrics.accuracy_score(y_train, y_pred_train).round(2))
print("Testing Accuracy:",metrics.accuracy_score(y_test, y_pred_test).round(2))

#cm = metrics.confusion_matrix(y_train, y_pred_train)
#print(cm)


# plotting the graph
# Plot Decision Region using mlxtend's awesome plotting function
# pip install mlxtend
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=X.values, 
                      y=y.values,
                      clf=clf, 
                      legend=4)




