# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 18:27:32 2022

@author: Hi
"""

import pandas as pd  
customer_data = pd.read_csv('D:\\My Classes\\ExcelR\\Data Science ExcelR\\Latest DS Material\\Day 17 - Clustering introduction, Hierarchical clustering\\shopping_data.csv', delimiter=',') 
customer_data.shape
customer_data.head()
X = customer_data.iloc[:, 3:5].values 
X.shape


import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete')) 

## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(X)

Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')  

Y_clust = pd.DataFrame(Y)
Y_clust[0].value_counts()
##############################################################################




