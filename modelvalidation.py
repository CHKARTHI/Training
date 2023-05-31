# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 14:32:13 2022

@author: Hi
"""

# step1: importing the data
import pandas as pd
df = pd.read_csv("Boston.csv")
df.shape

# step2: # Data cleannig, Data visualization
# boxplot
# bar graph
# histogram

df.info()
list(df)

# step3: split the data into X and Y variables
Y = df["medv"]
X = df.iloc[:,1:14]
list(X)

# step4: Data partition
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

#=============================================================
# step4: Model fitting with logistic regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

Y_pred_train = LR.predict(X_train)
Y_pred_test = LR.predict(X_test)

# step5: metrics 
from sklearn.metrics import mean_squared_error

train_error = mean_squared_error(Y_train,Y_pred_train)
print("Training Error:" ,train_error.round(2))

test_error = mean_squared_error(Y_test,Y_pred_test)
print("Test Error:" ,test_error.round(2))


#=============================================================
trainingerror = []
testerror = []

for i in range(1,501):
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=i)
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    trainingerror.append(mean_squared_error(Y_train,Y_pred_train))
    testerror.append(mean_squared_error(Y_test,Y_pred_test))
    
print(trainingerror)
print(testerror)


TE0 = pd.DataFrame(trainingerror)
TE1 = pd.DataFrame(testerror)

import numpy as np
np.mean(TE0)
np.mean(TE1)

TE1[0].hist()
TE1[0].describe()

#=============================================================



















    
    














