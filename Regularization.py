# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:37:51 2022

@author: Hi
"""
#pd.options.display.float_format = '{:.3f}'.format

# step1: importing the data
import numpy as np
import pandas as pd
df = pd.read_csv("Hitters_final.csv")
df.shape
list(df)

# step2: # Data cleannig, Data visualization
# boxplot
# bar graph
# histogram

# step3: split the data into X and Y variables
Y = df["Salary"]
X = df.iloc[:,1:17]
list(X)

# step4: scaling the data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)
X_scale

pd.DataFrame(X_scale).shape


# step5: Data partition
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X_scale,Y, test_size=0.3, random_state=42)


#=============================================================
# step6: Model fitting with linear regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

Y_pred_train = LR.predict(X_train)
Y_pred_test = LR.predict(X_test)

# step5: metrics 
from sklearn.metrics import mean_squared_error

train_error = mean_squared_error(Y_train,Y_pred_train)
print("Training Error:" ,np.sqrt(train_error).round(3))

test_error = mean_squared_error(Y_test,Y_pred_test)
print("Test Error:" ,np.sqrt(test_error).round(3))

#======================================================
############### BREAK ########################
#======================================================

from sklearn.linear_model import Ridge
RR  = Ridge(alpha=40)
RR.fit(X_train,Y_train)

Y_pred_train = RR.predict(X_train)
Y_pred_test = RR.predict(X_test)
train_error = mean_squared_error(Y_train,Y_pred_train)
print("Training Error:" ,np.sqrt(train_error).round(3))
test_error = mean_squared_error(Y_test,Y_pred_test)
print("Test Error:" ,np.sqrt(test_error).round(3))

RR.coef_

pd.DataFrame(RR.coef_)
pd.DataFrame(X.columns)
pd.concat([pd.DataFrame(RR.coef_),pd.DataFrame(X.columns)],axis=1)

#======================================================

from sklearn.linear_model import Lasso
LS  = Lasso(alpha=20)
LS.fit(X_train,Y_train)

Y_pred_train = LS.predict(X_train)
Y_pred_test = LS.predict(X_test)
train_error = mean_squared_error(Y_train,Y_pred_train)
print("Training Error:" ,np.sqrt(train_error).round(3))
test_error = mean_squared_error(Y_test,Y_pred_test)
print("Test Error:" ,np.sqrt(test_error).round(3))

pd.DataFrame(LS.coef_)
pd.DataFrame(X.columns)
pd.concat([pd.DataFrame(LS.coef_),pd.DataFrame(X.columns)],axis=1)

#======================================================


X_new = X.drop(X.columns[[0,2,3,6,7,9,12,14,15]], axis=1)
list(X_new)

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_new_scale = SS.fit_transform(X_new)
X_new_scale

pd.DataFrame(X_new_scale).shape

# step5: Data partition
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X_new_scale,Y, test_size=0.3)


# step6: Model fitting with linear regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)
Y_pred_train = LR.predict(X_train)
Y_pred_test = LR.predict(X_test)
train_error = mean_squared_error(Y_train,Y_pred_train)
print("Training Error:" ,np.sqrt(train_error).round(3))

test_error = mean_squared_error(Y_test,Y_pred_test)
print("Test Error:" ,np.sqrt(test_error).round(3))

# Evaluate using Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5,shuffle=False)
LR = LinearRegression()

results = abs(cross_val_score(LR, X_new_scale, Y, cv=kfold, scoring='neg_mean_squared_error'))
results
print("Test Error:" ,np.sqrt(np.mean(results)).round(3))



















