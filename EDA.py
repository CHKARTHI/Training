# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:21:04 2022

@author: Hi
"""

import pandas as pd
pd.options.display.float_format = '{:.3f}'.format

df = pd.read_csv("Callcenterdata.csv")

df.shape
list(df)
df.head()

# box plot
df.boxplot(column=['Calls Rec.'], vert=False)

import numpy as np
Q1 = np.percentile(df['Calls Rec.'],25)
Q3 = np.percentile(df['Calls Rec.'],75)

IQR = Q3 - Q1
LW = Q1 - (1.5*IQR)
UW = Q3 + (1.5*IQR)


df['Calls Rec.']<LW
df[df['Calls Rec.']<LW]

df['Calls Rec.']>UW
df[df['Calls Rec.']>UW]

df[(df['Calls Rec.']<LW) | (df['Calls Rec.']>UW)]
#==========================================================================
# histogram #
df = pd.read_csv("market.csv")
df.shape
df.head()

df.dtypes

df["Sales"].describe()

df["Sales"] = df.Sales.str.replace(",",'').astype(float)
df["Inventory"] = df.Inventory.str.replace(",",'').astype(float)
df["Returns"] = df.Returns.str.replace(",",'').astype(float)


df["Sales"].hist()
#==========================================================================

# covariance and correlation

df[['Sales','Returns']].cov()
df[['Sales','Returns']].corr()

#==========================================================================

# bar graph

t1 = df.groupby("Region").size()
t1.plot(kind='bar')

t2 = pd.crosstab(index=df['Product'], columns=df['Region'])
print(t2)
t2.plot(kind="bar",figsize=(8,4))
S





