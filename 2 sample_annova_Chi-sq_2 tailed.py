# # -*- coding: utf-8 -*-
# """
# Spyder Editor

# This is a temporary script file.
# """

# import pandas as pd
# import numpy as np
# import pandas


 x=pd.read_csv("market.csv")

# #H0 is both sates populations are not euql (u1=u2)
# #H1 is both sates populations are not euql (u1!=u2)

# x['Region'].value_counts()

 from scipy import stats

 x["Returns"]=x.Returns.str.replace(",",'').astype(float)

 x_new=x.pivot_table(index='Region',values="Returns")
 x_new


n1=31 #sample size eastern europe
n2=62 #sample size is western europe

 p1=2796.806452/x["Returns"].sum() # State1
 p2=2737.983871/x["Returns"].sum() #state 2

props = np.array([p1*100,p2*100])
sampsize=np.array([31,62])

 from statsmodels.stats.proportion import proportions_ztest

zcal, pval = proportions_ztest(props, sampsize)

alpha=0.05

 if pval < alpha:
   print("Ho is rejected and H1 is accepted")
 else:
     print("H1 is rejected and H0 is accepted")



#ANOVA
import pandas as pd

y = pd.read_csv("promots.csv")
y.shape
y[list]
list(y)

from scipy import stats
from statsmodels.formula.api import ols
lm1 = ols('sales ~ C(promotions)',data=y).fit()
import statsmodels.api as sm
table = sm.stats.anova_lm(lm1, type=1) # Type 1 ANOVA DataFrame

print(table)



# ....................



import scipy.stats as anovatable
rv1 = anovatable.f(dfn=3, dfd=16)
rv1.ppf(0.95).round(4)
F_table_value= 3.2389
Fcalc_value= 9.693587


if (Fcalc_value > F_table_value):
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    
   #-----Chi sq
   
   import pandas as pd
   df = pd.read_csv("credit_new.csv")
   df.shape
   
   list(df)

   df['Cards']
   df['Ethnicity']

   pd.crosstab(df['Cards'],df['Ethnicity'])
    pip install researchpy
    import researchpy as rp
    table, results = rp.crosstab(df['Cards'], df['Ethnicity'], test= 'chi-square')
    print(table)
    print(results)


alpha = 0.05
p = 0.0395  

# if p < alpha:
#     print("Ho is rejected and H1 is accepted")
# else:
#     print("H1 is rejected and H0 is accepted")

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    