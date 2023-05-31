# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 17:02:54 2022

@author: Hi
"""

# step1: importing the data
import pandas as pd
df = pd.read_csv("breast_cancer.csv")
df.shape

# step2: # Data cleannig, Data visualization
# boxplot
# bar graph
# histogram

df.corr()
list(df)

# step3: split the data into X and Y variables
# Data Transformation
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["Class"] = LE.fit_transform(df["Class"])

Y = df["Class"]
X = df.iloc[:,1:10]
list(X)

# step4: Data partition
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

print(X_train.shape)
print(X_test.shape)

print(Y_train.shape)
print(Y_test.shape)




























#=============================================================
# step4: Model fitting with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)

Y_pred_train = logreg.predict(X_train)
Y_pred_test = logreg.predict(X_test)

# step5: metrics 
from sklearn.metrics import accuracy_score

train_ac = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:" ,train_ac.round(2))

test_ac = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:" ,test_ac.round(2))

#=============================================================

#=============================================================
# step4: Model fitting with KNN classifier
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5)

KNN.fit(X_train,Y_train)

Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

# step5: metrics 
from sklearn.metrics import accuracy_score

train_ac = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:" ,train_ac.round(2))

test_ac = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:" ,test_ac.round(2))

# we have to try different k values and check the test accuracies
# for which k value we are getting the higheste accuracy, that k value is final










