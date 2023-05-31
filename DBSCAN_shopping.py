"""
Created on Fri Jun 24 15:25:21 2022

"""

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


