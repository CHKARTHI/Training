"""
Created on Sat Apr  2 20:11:06 2022
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Movie.csv')
df.shape
df.head()

df.sort_values('userId')

#number of unique users in the dataset
len(df)
len(df.userId.unique())

df['rating'].value_counts()
df['rating'].hist()


len(df.movie.unique())

df.movie.value_counts()

user_df = df.pivot(index='userId',
                                 columns='movie',
                                 values='rating')

user_df
user_df.iloc[0:0,1]
user_df.iloc[200]
list(user_df)

#Impute those NaNs with 0 values
user_df.fillna(0, inplace=True)

user_df.shape

# from scipy.spatial.distance import cosine correlation
# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances

user_sim = 1 - pairwise_distances(user_df.values,metric='cosine')

#user_sim = 1 - pairwise_distances( user_df.values,metric='correlation')

user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_sim_df.index   = df.userId.unique()
user_sim_df.columns = df.userId.unique()

user_sim_df.iloc[0:5, 0:5]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

#Most Similar Users
user_sim_df.max()

user_sim_df.idxmax(axis=1)[0:10]

df[(df['userId']==6) | (df['userId']==168)]

user_6=df[df['userId']==6]
user_168=df[df['userId']==168]


user_3=df[df['userId']==3]
user_11=df[df['userId']==11]

df[(df['userId']==3) | (df['userId']==11)]

pd.merge(user_3,user_11,on='movie',how='inner')
pd.merge(user_3,user_11,on='movie',how='outer')


#-------------------------------------------------------------------




