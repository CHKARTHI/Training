# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 14:28:57 2022

@author: DELL
"""

import pandas as pd
import numpy as np

x=pd.read_csv("Anydomain.csv")


import matplotlib.pyplot as plt

x.corr()

#X1=x[['X4']]
#X1=x[['X4','X1']]
#X1=x[['X4','X3']]
X1=x[['X1','X2','X3','X4']]
X1
X1.dropna(inplace=True)
X2=x['Y']
X2
X2.dropna(inplace=True)
plt.scatter(x["X1"],x["Y"], color='black')
plt.scatter(x["X2"],x["Y"], color='black')

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X1,X2)
lr.intercept_
lr.coef_

X2_pred=lr.predict(X1)

# plt.scatter(X1,X2_pred, color='black')
# plt.plot(X1,X2_pred, color='red')



from sklearn.metrics import mean_squared_error
mse = mean_squared_error(X2,X2_pred)
print("Mean square error: ",mse.round(2))


import numpy as np
RMSE = np.sqrt(mse)
print("Root mean square error: ",RMSE.round(2))

from sklearn.metrics import mean_squared_error, r2_score
r2=r2_score(X2,X2_pred)
print('RMSE',RMSE)
print('r2',r2)

#---------------
pip install statsmodels 
import statsmodels.api as sm
X_new=sm.add_constant(X1)
X_new
lm2=sm.OLS(X2,X_new).fit()
lm2.summary()











