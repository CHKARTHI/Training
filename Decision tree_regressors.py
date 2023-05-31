# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:19:14 2022

@author: DELL
"""


#=======================================================================
import pandas as pd
df = pd.read_csv("Boston.csv")  
df.shape
df.info()

# # Label encode
# from sklearn.preprocessing import LabelEncoder 
# LE = LabelEncoder()
# df['ShelveLoc'] = LE.fit_transform(df['ShelveLoc'])
# df['Urban'] = LE.fit_transform(df['Urban'])
# df['US'] = LE.fit_transform(df['US'])
# df.head()


X = df.iloc[:,1:13]
Y = df['medv']

# split the data in to train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)


# model fitting
from sklearn.tree import DecisionTreeRegressor  
reg = DecisionTreeRegressor(max_depth=(7))
reg.fit(X_train,Y_train)


reg.tree_.max_depth # number of levels
reg.tree_.node_count # counting the number of nodes

Y_pred_train = reg.predict(X_train)
Y_pred_test = reg.predict(X_test)

from sklearn.metrics import mean_squared_error
MSE_Train = mean_squared_error(Y_train,Y_pred_train).round(2)
MSE_Test = mean_squared_error(Y_test,Y_pred_test).round(2)

import numpy as np
RMSE_Train=np.sqrt(MSE_Train)
RMSE_Test=np.sqrt(MSE_Test)
RMSE_Train
RMSE_Test

#######################################################

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
Regressor=DecisionTreeRegressor(max_depth=10)

Bag=BaggingRegressor(base_estimator=Regressor,max_samples=0.6,n_estimators=100)

Bag.fit(X,Y)
Y_pred_bag=Bag.predict(X)
Bagging_error = mean_squared_error(Y,Y_pred_bag)


############################################################

from sklearn.ensemble import RandomForestRegressor
Regressor=RandomForestRegressor(max_depth=10)
RF=RandomForestRegressor(max_features=0.4,n_estimators=500)
RF.fit(X,Y)
Y_pred_RF=RF.predict(X)
RF_error = mean_squared_error(Y,Y_pred_RF)

##########################################################
from sklearn.ensemble import GradientBoostingRegressor
GB = GradientBoostingRegressor(n_estimators=500,learning_rate=0.1)

GB.fit(X,Y)
Y_pred_GB = GB.predict(X)
GB_error = mean_squared_error(Y,Y_pred_GB)



















































