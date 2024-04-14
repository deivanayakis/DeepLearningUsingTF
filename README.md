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

# c) Preprocess the Dataset.

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

Each data in the dataset have different images which may vary in their size so to make it compatible for the deep learning model, we use reshape command to reshape all the data as same dimension

X_train = X_train.astype('float32') / 255.0 
X_test = X_test.astype('float32') / 255.0

This data may be in the form of integer , if we apply this integer values of image data to the model then model consider the small deviation as big so conversion of integer to float is needed.
It is divided by 255 (image can have 0-255)  , if we do like this all data fall in the range of 0-1 hence it is easier for the model to learn. This process is called "Scaling" or "Normalization".





