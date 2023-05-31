# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 14:21:17 2022

@author: Hi
"""

#-----------------------------------------------------------------------
import numpy as np
    
n1=247 # sample size 1
n2=308 # sample size 2
alpha = 0.05

p1 = 0.37 # state1
p2 = 0.39 # state2 
round(p1*100)
round(p2*100)
    
props = np.array([p1*100, p2*100])
sampsize = np.array([247, 308])

from statsmodels.stats.proportion import proportions_ztest
stat, pval = proportions_ztest(props, sampsize)

if pval < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    
#========================================================


import pandas
df = pd.read_csv("market.csv")
df

df['Region'].value_counts()

df_new = df.pivot_table(index='Region',values="Returns")

n1=31 # sample size Eastern Europe
n2=62 # sample size Western Europe
alpha = 0.05

p1 = 2796.806452/df["Returns"].sum() # state1
p2 = 2737.983871/df["Returns"].sum() # state1

props = np.array([p1*100, p2*100])
sampsize = np.array([31, 62])

from statsmodels.stats.proportion import proportions_ztest
stat, pval = proportions_ztest(props, sampsize)

if pval < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

### ANOVA


























