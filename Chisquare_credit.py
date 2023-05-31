"""
Created on Thu May 26 16:13:54 2022

"""

import pandas as pd
df = pd.read_csv("credit_new.csv")
df.shape

list(df)

df['Cards']
df['Ethnicity']

pd.crosstab(df['Cards'],df['Ethnicity'])

#---------------------------------------------------------------
# pip install researchpy
import researchpy as rp
table, results = rp.crosstab(df['Cards'], df['Ethnicity'], test= 'chi-square')

print(table)
print(results)

alpha = 0.05
p = 0.0395
if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")




#---------------------------------------------------------------
# Chi square table values for given alpha and degrees of freedom
import scipy.stats as stats
chi_table = stats.chi2.ppf(q = 0.95, df = 8)
chi_table.round(4)







