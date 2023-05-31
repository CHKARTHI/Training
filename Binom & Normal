# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 14:16:58 2022

@author: DELL
"""

from scipy import stats

#binominal for decret

bi = stats.binom(5,0.6) #n,p
bi
bi.pmf(3)  #3 events

bi.pmf(3).round(2)

bi.pmf([0,1,2]).sum()

#atmost 2--P(x=0)+p(x=2)+p(x=3)
1-bi.cdf(2).round(3)

bi.pmf([3,4,5]).sum() #1-above case

x=stats.binom(250,0.7)
1-x.cdf(159)
x.cdf(159)

# Normal for continues

# Z distribution, standardization

x=stats.norm(170,10)  #mean, SD
x

#p(x<170)

x.cdf(170) #0 to 170

#p(160<x<180)

x.cdf(180)-x.cdf(160)
x.cdf(200)-x.cdf(140)

x=stats.norm(29,5)
x.cdf(34)-x.cdf(30)
1-x.cdf(23)
1-x.cdf(40)

x=stats.norm(90,5)
1-x.cdf(100)




























