# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:08:20 2022

@author: Hi
"""

import pandas as pd
df = pd.read_csv("Cars_100.csv")
df

df["USCARS"].mean()
df["GERMANCARS"].mean()


from scipy import stats

zcalc ,pval = stats.ttest_ind( df["USCARS"] , df["GERMANCARS"] ) 

print("Zcalcualted value is ",zcalc.round(4))
print("P-value is ",pval.round(4))

if pval<0.05:
    print("reject null hypothesis, Accept Alternative hypothesis")
else:
    print("accept null hypothesis, Reject Alternative hypothesis")

