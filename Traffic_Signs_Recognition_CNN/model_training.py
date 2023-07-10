import matplotlib.image as mpimg
import os
import random as r

from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation, RandomZoom
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob
import cv2

import warnings
warnings.filterwarnings('ignore')


# Data Extracting
from zipfile import ZipFile

data_path = 'traffic-sign-dataset-classification.zip'
ZipFile(data_path).extractall()
print('Data Extracting completed')


# Data Visualization
dataset = 'traffic_Data/DATA'

labelfile = pd.read_csv('labels.csv')

img = cv2.imread('traffic_Data/DATA/0/000_1_0024.png')
#plt.imshow(img)
print('Data Visualization is completed')


# Data Preproccessing
train_ds = image_dataset_from_directory(dataset,
                                        validation_split= 0.2,
                                        subset= 'training',
                                        image_size=(224, 224),
                                        seed=123,
                                        batch_size=32)

val_ds = image_dataset_from_directory(dataset,
                                        validation_split= 0.2,
                                        subset= 'validation',
                                        image_size=(224, 224),
                                        seed=123,
                                        batch_size=32)

class_numbers = train_ds.class_names
class_names = []

for i in class_numbers:
    class_names.append(labelfile['Name'][int(i)])
print('Data Preproccessing completed')


# Data Augmentation
data_augmentation = Sequential([
    RandomFlip('horizontal', input_shape= (224,224,3)),
    RandomRotation(0.1),
    RandomZoom(0.2),
    RandomFlip(mode= 'horizontal_and_vertical')
])
print('Data Augmentation completed')


# Model Architecture
model = Sequential()

model.add(data_augmentation)
model.add(Rescaling(1./255))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dense(len(labelfile), activation='softmax'))

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer= 'adam', metrics= ['accuracy'])
print('Model Architecture completed')


# Model Training
m_callbacks = [ModelCheckpoint('trained_model',
                               monitor= 'val_accuracy',
                               verbose= 1,
                               save_best_only= True,
                               save_weights_only= True)]

history = model.fit(train_ds, validation_data=val_ds, epochs=60, callbacks=m_callbacks, verbose=1)
print('Model Training completed')
