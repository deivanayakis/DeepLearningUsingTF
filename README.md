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

Droput Layer - Including the dropout layer helps to prevent overfitting, it randomly drops the neuron along with their connections
Dense Layer - Each neuron of dense layer receives input from all the neurons of previous layer and it is used to classify the image based on output from previous layer.

# e) Model Compilation

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Compiling the model configuring the learning process of model, it should be done before training the model. It takes metrics argument such loss function, optimizer and metrics.
Loss function tells how good a neural network model is in performing a certain task. categorical_crossentropy is used for multiclass classification.
Adam optimizer means Adaptive moment estimation, it is an iterative optimization algorithm which is used to minimize the loss function while training the model.
Accuracy metrics calculates how model equals prediction with actual labels.

# f) Training the Model

model.fit(X_train, y_train, epochs=20,batch_size=64)

epoch means number of times the training data is used for train the model and batch size refers number of samples processed.

# g) Evaluating the Model

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

Trained Model is evaluated by applying the testing dataset.

# h) Predict the label for given image

from PIL import Image
img = Image.open('/content/7.jpg')
img = img.resize((28, 28))
img = img.convert('L')
img_array = np.array(img)
img_array = np.reshape(img_array, (1, 28, 28, 1))
img_array = img_array.astype('float32') / 255.0
pred = model.predict(img_array)
predclass = np.argmax(pred)
print("Predicted class:", predclass)

Input image is read by specifing the path of that image and preprocess the image. Apply the preprocesses image numpy array to the model and predict the respective label.


# 2) Sentiment Analysis using TF_IDF encoding

# a) Load the data 

import pandas as pd
df = pd.read_csv('/content/data.csv', encoding='iso-8859-1')
df.head()

# b) Preprocessing the text data

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nltk - Natural Language Toolkit (NLTK) for natural language processing tasks.
nltk.corpus.stopwords - NLTK's stopwords corpus for removing common words that do not contribute much to text meaning.
nltk.tokenize.word_tokenize - NLTK's word_tokenize function for splitting text into words or tokens.
nltk.stem.WordNetLemmatizer - NLTK's WordNetLemmatizer for reducing words to their base or root form.

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        preprocessed_text = ' '.join(tokens)
    else:
        preprocessed_text = ''
    return preprocessed_text

df['text'] = df['text'].apply(preprocess_text)
df.head()

# c) Construct the model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(max_features=1000)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)

Split the data, apply the tfidf vectorization and build the model using Logistic regression.

# d) Predict for the new text

txt = preprocess_text("It is a Nice product!!")
txt = tfidf.transform([txt])
predicted_sentiment = model.predict(txt)
print("Predicted Sentiment:", predicted_sentiment)

O/p : Predicted Sentiment: ['positive']

     

