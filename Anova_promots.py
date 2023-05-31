# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:55:24 2022

@author: Hi
"""

import pandas as pd
df = pd.read_csv("promots.csv")
df.shape
list(df)


from statsmodels.formula.api import ols
lm1 = ols('sales ~ C(promotions)',data=df).fit()

import statsmodels.api as sm
table = sm.stats.anova_lm(lm1, type=1) # Type 1 ANOVA DataFrame

print(table)

p = 0.000695

if p < 0.05:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")


#==================================================    
#==== FTABLE VALUE ================================
import scipy.stats as anovatable
rv1 = anovatable.f(dfn=3, dfd=16)
rv1.ppf(0.95).round(4)

F_table_value= 3.2389
Fcalc_value= 9.693587

if (Fcalc_value > F_table_value):
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    
#==================================================

import pandas as pd
df = pd.read_csv("credit_new.csv")
df.shape























