# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 14:03:17 2022

@author: DELL
"""

import pandas as pd
x=pd.read_csv("Callcenterdata.csv")
x.shape
list(x)
x.head()

#box plot
x.boxplot(column=['Calls Rec.'],vert=False)
x
import numpy as np
q1=np.percentile(x['Calls Rec.'], 25)
q3=np.percentile(x['Calls Rec.'], 75)
iqr=q3-q1

lw=q1-(1.5*iqr)
uw=q3+(1.5*iqr)

x['Calls Rec.']<lw
x['Calls Rec.']>lw

x[(x['Calls Rec.']<lw) | (x['Calls Rec.']>lw)]

#histogram

y=pd.read_csv("market.csv")
y.shape
y.dtypes
y["Sales"].describe()
y["Sales"]=y.Sales.str.replace(",",'').astype(float)
y["Sales"].dtypes
y["Inventory"]=y.Inventory.str.replace(",",'').astype(float)
y["Returns"]=y.Returns.str.replace(",",'').astype(float)


#Cov & Corelations
y.Sales.hist()
y[['Sales','Returns']].cov()
y[['Sales','Returns']].corr()

#bargraph

t1=y.groupby("Region").size()
t1.plot(kind='bar')
#([index x=y['Products'],index y=y['Sales']])

t2=pd.crosstab(index=y['Product'], columns=[y['Sales'],y['Returns']])
t2

t2.to_csv(r'C:\Users\DELL\Desktop\Data Science\Python\test.csv')
t2.plot(kind="bar",figsize=(8,4))
t2

pd.options.display.float_format = '{:.3f}'.format #remove scentific

#from scipy import stats

from statsmodels import stats


from statsmodels.stats import weightstats as ztests

x=pd.read_csv("Lungcapdata.csv")

zcal,pval = ztests.ztest(x['LungCap'],value=8,alternative='smaller')

print("zcalculated value is", zcal)
print("p-valuese is", pval)

if pval<0.05:
    print('rejecr null,accept alternative')

else :
    print('rejecr Alt,accept null')

help(ztests.ztest)

help(ztests)
























