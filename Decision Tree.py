# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 16:33:49 2022

@author: excel
"""

import pandas as pd  

df = pd.read_csv("Cricket.csv")  
df.shape
list(df)
df.head()


# Label encode
from sklearn.preprocessing import LabelEncoder 
LE = LabelEncoder()
df['Gender'] = LE.fit_transform(df['Gender'])
df['Class'] = LE.fit_transform(df['Class'])
df.head()

# split the variable
X = df.drop('Cricket', axis=1)  #Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
Y = df['Cricket']

# model fitting
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier() 
classifier.fit(X,Y)
Y_pred = classifier.predict(X)

from sklearn.metrics import accuracy_score
accuracy_score(Y,Y_pred).round(2)

import numpy as np
X_a = np.array([[1,1]])
classifier.predict(X_a)

################### ONE HOT ENCODING #####################

df = pd.read_csv("Cricket.csv")  
df.shape

# Label encode
from sklearn.preprocessing import OneHotEncoder 
OHE = OneHotEncoder()
X_gender = OHE.fit_transform(df[['Gender']]).toarray()


















