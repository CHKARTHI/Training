# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:34:52 2022

@author: DELL
"""

#Central limmit theorem

from scipy import stats
import numpy as np

x=np.random.normal(100,50,100)
x
import pandas as pd
y=pd.DataFrame(x)
y

y.hist()
# H0 : Datais normal
#H1: data is not normal

from scipy import stats
from scipy.stats import shapiro

stat, pval = shapiro(y) 
pval
#H0 accepted. 

##linear regression-1

x=pd.read_csv("Viralcount_Drug.csv")

#scatterplot-2

import matplotlib.pyplot 
plt.scatter(x["drug"],x["Viralcount"],color='black')
plt.show()
x.corr()


#split variables in to two parts-3
x1=x[["drug"]]
x2=x["Viralcount"]
x1
x2
#Model fitting - 4

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x1,x2)
lr


lr.intercept_
lr.coef_

#Preddiction-5

x2_pred=lr.predict(x1)
x2_pred

plt.scatter(x["drug"],x2_pred,color='red')
plt.scatter(x["drug"],x["Viralcount"],color='black')
plt.show()
x.corr()
plt.plot(x["drug"],x2_pred,color='red')
plt.scatter(x["drug"],x["Viralcount"],color='black')


#Metrics - error

from sklearn.metrics import mean_squared_error as mse

mse(x2,x2_pred)

import numpy as np
RMSE=np.sqrt(mse(x2,x2_pred))
RMSE.round(2)


#multiple linear regression

x=pd.read_csv('Advertising.csv')
plt.scatter(x["TV"],x["sales"],color='black')
plt.show()
x.corr()

plt.scatter(x["radio"],x["sales"],color='red')
plt.scatter(x["newspaper"],x["sales"],color='green')

x1=x[["TV"]]
x2=x[["radio"]]
x3=x[["newspaper"]]
x4=x["sales"]


lr1=LinearRegression()
lr1.fit(x1,x4)
lr1
lr1.intercept_
lr1.coef_

lr2=LinearRegression()
lr2.fit(x2,x4)
lr2
lr2.intercept_
lr2.coef_

lr3=LinearRegression()
lr3.fit(x3,x4)
lr3
lr3.intercept_
lr3.coef_

x4_pred=lr.predict(x1)
y1=x4_pred

x4_pred=lr.predict(x2)
y2=x4_pred

x4_pred=lr.predict(x3)
y3=x4_pred
y3


plt.scatter(x1,y1)
plt.scatter(x2,y2,color='black')
plt.show()
x.corr()
plt.plot(x["drug"],x2_pred,color='red')
plt.scatter(x3,y3,color='black')


mse(x1,y1)

RMSE=np.sqrt(mse(x1,y1))
RMSE.round(2)


mse(x2,y1)

RMSE=np.sqrt(mse(x2,y1))
RMSE.round(2)


mse(x4,y1)

RMSE=np.sqrt(mse(x4,y1))
RMSE.round(2)


......................................................
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
X = df[['TV','radio']]
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













































































