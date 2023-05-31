# step1: import the data
import pandas as pd
df = pd.read_csv("Advertising.csv")
df.shape
df

#=========================================
# identify the relationships between X and Y variables
# step2: # scatter plot
import matplotlib.pyplot as plt
plt.scatter(df["TV"], df["sales"], color='black')

plt.show()

plt.scatter(df["radio"], df["sales"], color='black')
plt.show()

plt.scatter(df["newspaper"], df["sales"], color='black')
plt.show()

df.corr()

#=========================================
# step3: 
# split the variables in to two parts
X = df[['TV','radio','newspaper']]
Y = df["sales"]

#=========================================
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

# prediction line in scatter plot
# import matplotlib.pyplot as plt
# plt.scatter(X, Y, color='black')
# plt.plot(X,Y_pred, color='Red')
# plt.show()
#=========================================
# step6: Metrics

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean square error: ",mse.round(2))

import numpy as np
RMSE = np.sqrt(mse)
print("Root mean square error: ",RMSE.round(2))

#=========================================
