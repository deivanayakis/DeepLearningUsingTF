# DeepLearningUsingTF

# 1. Mnist Image Classication

# a) Import Needed libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D, Flatten, Dropout

Numpy - For doing numerical operation on arrays, matrices etc.
Pandas - For data handling, manipulation and analysis
TensorFlow - To build and Train the deep learning models
Keras - Neural Network API for building and training the deep learning models easily.
Sequential - Helps to create linear stack of layers (one by one)
Dense, Conv2D, MaxPooling2D, Flatten, Dropout - Layers present in the neural network

# b) Load the Mnist (contains Handwritten digits Image Dataset) dataset and split into train and test.

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

Data is loaded and splitted into train and test. Training data is applied when the model is training, for the prediction and accuracy calculation testing data can be applied and compare the results.

# c) Preprocess the Dataset.

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

Each data in the dataset have different images which may vary in their size so to make it compatible for the deep learning model, we use reshape command to reshape all the data as same dimension

X_train = X_train.astype('float32') / 255.0 
X_test = X_test.astype('float32') / 255.0

This data may be in the form of integer , if we apply this integer values of image data to the model then model consider the small deviation as big so conversion of integer to float is needed.
It is divided by 255 (image can have 0-255)  , if we do like this all data fall in the range of 0-1 hence it is easier for the model to learn. This process is called "Scaling" or "Normalization".

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

It converts the labels into one hot encoded vector which is widely used for multiclass classification, since mnist dataset contains 10 classess from 0 to 9 it is used.The value 1 is placed at the index corresponding to the class label, and all other elements in the vector are set to 0.
if y_train[0] is 5, after applying to_categorical, it will be represented as [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

# d) Define Model Architecture

model = Sequential()

It allows us to easily build neural networks by adding layers one by one in sequence i.e help to create linear stack of layers

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

The convolutional layers extract features from the input images, and the max pooling layers reduce spatial dimensions by retaining the most important information (corners, shapes, edges, etc..).

model.add(Flatten())

It reshapes the multi-dimensional input into a one-dimensional array. This is important because following layers like Dense need one dimensional input.

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

