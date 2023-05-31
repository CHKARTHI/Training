"""
Created on Sat Aug 27 14:25:48 2022
"""

import pandas as pd
df = pd.read_csv("Anydomain.csv")
df.shape
df

df.corr()


# X = df[["X4"]]
X = df[["X4","X1"]]
# X = df[["X4","X3"]]
Y = df["Y"]

#=====================================================
# step4: 
# model fitting
from sklearn.linear_model import LinearRegression
LR =LinearRegression()
LR.fit(X,Y)

# Bias value
LR.intercept_

# coeffcient value
LR.coef_

#=========================================
# step5: 

# predictions
Y_pred = LR.predict(X)

#=========================================
# step6: Metrics

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,Y_pred)
print("Mean square error: ",mse.round(2))

import numpy as np
RMSE = np.sqrt(mse)
print("Root mean square error: ",RMSE.round(2))

r2 = r2_score(Y,Y_pred)
print("R square: ",r2.round(3))

#=========================================

pip install statsmodels
import statsmodels.api as sma
X_new = sma.add_constant(X)
lm2 = sma.OLS(Y,X_new).fit()
lm2.summary()











