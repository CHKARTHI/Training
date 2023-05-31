# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:37:31 2022

@author: excel
"""
import pandas as pd

df = pd.read_csv("Boston.csv")  
df.shape
df.head() 
df.describe()
type(df)
df.info()

# split the variables as X and Y
X = df.iloc[:,1:14]  #Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)
Y = df['medv']

###############################################################

from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30) 

from sklearn.tree import DecisionTreeRegressor
Regressor = DecisionTreeRegressor(max_depth=(10))

Regressor.fit(X_train, Y_train)
Y_pred_train = Regressor.predict(X_train) 
Y_pred_test = Regressor.predict(X_test) 

Regressor.tree_.node_count # counting the number of nodes
Regressor.tree_.max_depth # number of levels

from sklearn.metrics import mean_squared_error

Training_error = mean_squared_error(Y_train,Y_pred_train)
Test_error = mean_squared_error(Y_test,Y_pred_test)

print("Training_error: ",Training_error.round(2))
print("Test_error: ",Test_error.round(2))

###############################################################

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
Regressor = DecisionTreeRegressor(max_depth=10)

Bag = BaggingRegressor(base_estimator=Regressor,max_samples=0.6,n_estimators=500)

Bag.fit(X,Y)
Y_pred_bag = Bag.predict(X)
Bagggin_error = mean_squared_error(Y,Y_pred_bag)
print("Bagggin_error: ",Bagggin_error.round(2))

###############################################################

from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(max_features=0.4,n_estimators=500)

RF.fit(X,Y)
Y_pred_bag = RF.predict(X)
RF_error = mean_squared_error(Y,Y_pred_bag)
print("Random Forests error: ",RF_error.round(2))

###############################################################

from sklearn.ensemble import GradientBoostingRegressor
GB = GradientBoostingRegressor(n_estimators=500,learning_rate=0.1)

GB.fit(X,Y)
Y_pred_bag = GB.predict(X)
GB_error = mean_squared_error(Y,Y_pred_bag)
print("Gradient Boosting error: ",GB_error.round(2))

###############################################################


from sklearn.ensemble import AdaBoostRegressor
AB = AdaBoostRegressor(n_estimators=1000,learning_rate=0.1)

AB.fit(X,Y)
Y_pred_bag = AB.predict(X)
AB_error = mean_squared_error(Y,Y_pred_bag)
print("AdaBoost Regressor error: ",AB_error.round(2))
###############################################################























