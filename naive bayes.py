"""
Created on Sun Sep  4 16:26:19 2022
"""

#======================================================================
import pandas as pd
df = pd.read_csv("mushroom.csv")
df.shape
list(df)
df.head()

# lable encode
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for eachcolumn in range(0,23):
    df.iloc[:,eachcolumn] = LE.fit_transform(df.iloc[:,eachcolumn])

df.head()

# split as X and Y vairables
X = df.iloc[:,1:]
Y  = df['Typeofmushroom']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, stratify=Y) # test_size=0.25

#======================================================================
# model development
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

Y_pred_train = MNB.predict(X_train)
Y_pred_test = MNB.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_train,Y_pred_train).round(2)
print("naive bayes model accuracy score:" , acc)

acc2 = accuracy_score(Y_test,Y_pred_test).round(2)
print("naive bayes model accuracy score:" , acc2)

