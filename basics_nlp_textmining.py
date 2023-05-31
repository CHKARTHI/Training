# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:54:01 2022

@author: excel
"""

#============ TOKENIZATION USING SPACY  =======================================

# install in anaconda prompt
# pip install spacy
#python -m spacy download en
import spacy
nlp = spacy.load("en_core_web_sm")


doc2 = nlp("Tesla isn't looking into startups anymore.")
for token in doc2:
    print(token.text)
    
#============ TOKENIZATION USING NLTK  =======================================

import nltk
nltk.download()

from nltk.tokenize import word_tokenize
text = "Hello, How are you? Are you available to come to interview?"
print(word_tokenize(text))

#=============================================================

# stemming
from nltk.stem import PorterStemmer
p_stemmer = PorterStemmer()
e_words= ["wait", "waiting", "waited", "waits"]

for word in e_words:
    print(word+' --> '+p_stemmer.stem(word))


from nltk.stem.snowball import SnowballStemmer
s_stemmer = SnowballStemmer(language='english')

words = ['run','runner','running','ran','runs']

for word in words:
    print(word+' --> '+s_stemmer.stem(word))

#=============================================================
# Lemmatization
from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"

for token in text.split():
    print(f"{token:{10}} --> {Lemmatizer.lemmatize(token)}")

#=============================================================
# stopwords
# Step 1 - Import nltk and download stopwords, and then import stopwords from NLTK
import nltk
from nltk.corpus import stopwords

print(stopwords.words('english'))
len(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

# Step 3 - Create a Simple sentence
text = "the city is beautiful, but due to traffic noice polution is increasing on daily basis which is hurting all the people"

# Step 4 - download and import the tokenizer from nltk
from nltk.tokenize import word_tokenize 
word_t = word_tokenize(text)
len(word_t)

newset = []
for x in word_t:
    if not x in stop_words:
        newset.append(x)

print(newset)
len(newset)









