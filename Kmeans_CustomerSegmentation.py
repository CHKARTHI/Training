# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:51:17 2020

@author: Hi
"""

import pandas as pd
df = pd.read_csv("shopping_data.csv")
df.shape
list(df)

df = df.iloc[:,2:]
df.shape

%matplotlib qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])
plt.show()

#
from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters = 5, n_init=30)
KMeans.fit(df)
Y = KMeans.predict(df)
Y = pd.DataFrame(Y)
Y[0].value_counts()

C = KMeans.cluster_centers_
C[:,0]
C[:,1]
C[:,2]

KMeans.inertia_


%matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])
ax.scatter(C[:,0],C[:,1],C[:,2], marker='*',c='Red',s=1000)
plt.show()


# by plotting elbow method we can decide which k is best
from sklearn.cluster import KMeans

inertia = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(df)
    inertia.append(km.inertia_)
    
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()


# new data with inertia and k values
df_new = pd.concat([pd.DataFrame(inertia,columns=["variance"]),pd.DataFrame(range(1, 11),columns=["k_value"])],axis=1)

import seaborn as sns
sns.barplot(x='k_value',y="variance", data=df_new, color="c");

##########################################################################3

#DBSCAN
#Import the libraries
import pandas as pd

# Import .csv file and convert it to a DataFrame object
df = pd.read_csv("shopping_data.csv")

print(df.head())


df.shape

print(df.info())

df.drop(['CustomerID','Genre'],axis=1,inplace=True)
array=df.values
array

from sklearn.preprocessing import StandardScaler
stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)
X

from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(X)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl
print(cl['cluster'].value_counts())

clustered = pd.concat([df,cl],axis=1)

noisedata = clustered[clustered['cluster']==-1]
print(noisedata)
finaldata = clustered[clustered['cluster']==0]














