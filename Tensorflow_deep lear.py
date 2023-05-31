# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:49:36 2022

@author: DELL
"""

#Deep learning_gradient decent techniques..

import pandas as pd
df=pd.read_csv("pima-indians-diabetes.csv",delimiter=",")

x=df.iloc[:,0:8]
y=df.iloc[:,8]
#pip install tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# create model
import numpy as np
model = Sequential()
model.add(Dense(12, input_dim=8,  activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(x, y, validation_split=0.30, epochs=250, batch_size=100)

# evaluate the model
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
history.history.keys()
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
