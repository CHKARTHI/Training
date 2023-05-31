"""
Created on Fri Jun 11 18:46:52 2021
"""
#===============================================================================================
# Set your directories
train_dir = 'D:\\CARRER\\My_Course\\Data Science Classes\\4 Module\\3 DEEP LEARNING\\03 CNN\\catsanddogs\\train\\'
test_dir = 'D:\\CARRER\\My_Course\\Data Science Classes\\4 Module\\3 DEEP LEARNING\\03 CNN\\catsanddogs\\test\\'

import os
os.listdir(train_dir)
for i in os.listdir(train_dir):
    print(i)

train_dogs = ['train_dir{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]  #get dog images
train_cats = ['train_dir{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]  #get cat images

test_imgs = ['test_dir{}'.format(i) for i in os.listdir(test_dir)] #get test images

#===============================================================================================
# The train data contains 25,000 images of both dogs and cats. 
# We are going to sample a small portion of the data for training because of memory and Ram limits. So therefore, we will use Data Augmentation to reduce time.

import random
train_imgs = train_dogs[:2000] + train_cats[:2000]  # slice the dataset and use 2000 in each class
len(train_imgs)
random.shuffle(train_imgs)  # shuffle it randomly

#===============================================================================================


#Lets declare our image dimensions
#we are using coloured images. 
nrows = 150
ncolumns = 150
channels = 3  #change to 1 if you want to use grayscale image

#pip install opencv-python
import cv2
print(cv2.__version__)
#A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images):
    X = [] # images
    y = [] # labels
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
        #get the labels        
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
    return X, y

#get the train and label data
X, y = read_and_process_image(train_imgs)

# Error occuring at above line at cv2.resize in function




#===============================================================================================
#Convert list to numpy array
import numpy as np
X = np.array(X)
y = np.array(y)


#Lets split the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)



#===============================================================================================
#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32

from tensorflow.keras import models, layers, optimizers

#from keras.preprocessing.image import img_to_array, load_img

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  #Sigmoid function at the end because we have just two classes


#Lets see our model
model.summary()

#===============================================================================================
#We'll use the RMSprop optimizer with a learning rate of 0.0001
#We'll use binary_crossentropy loss because its a binary classification
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

#Lets create the augmentation configuration
#This helps prevent overfitting, since we are using a small dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale



#Create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


#The training part #epochs=64 originally
#We train for 64 epochs with about 100 steps per epoch
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=1,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)

#===============================================================================================

