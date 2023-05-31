"""
Created on Sun Oct 16 14:35:55 2022
"""

import pandas as pd
df = pd.read_csv("smsspamcollection.tsv", sep='\t')
df.head()
df.shape

Y = df['label']
X = df['message']

Y
X

# Text preprocesing

#------------------------------------------------------
# Pre-Processing
df['message'] = df.message.map(lambda x : x.lower())
#------------------------------------------------------
# stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
for i in df['message'].index:
    df['message'].iloc[i] = stemmer.stem(df['message'].iloc[i])
#------------------------------------------------------
# Lemmatizer 
from nltk.stem import WordNetLemmatizer
Lemm = WordNetLemmatizer()
for x in df['message'].index:
    df['message'].iloc[x] = Lemm.lemmatize(df['message'].iloc[x])
#------------------------------------------------------
# stop words


#------------------------------------------------------
# feature extraction
# TOKENIZATION
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer()
Vt = Vectorizer.fit_transform(df['message'])
Vt.toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer().fit(Vt)
X_vect = transformer.transform(Vt)  
X_vect.shape
x_temp = X_vect.toarray()

# convering to pandas and exporting to csv
x_temp = pd.DataFrame(x_temp)
x_temp.to_csv("temp.csv")

#------------------------------------------------------

# Data partition
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X_vect,Y)

# naive baye
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,Y_train)
Y_pred = nb.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
cm

score = accuracy_score(Y_test,Y_pred)
score.round(2)
#===================================================================





