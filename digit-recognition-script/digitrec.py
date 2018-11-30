# Adapted from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

# imports
import gzip
import keras as kr
import numpy as np
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# Start neural network
model = kr.models.Sequential()

#Build our neural network model
model = Sequential()

#Build our neural network model
model = Sequential()

# Neural Network with 3 layers (1000, 750, 512)
model.add(kr.layers.Dense(units=1000, activation='relu', input_dim=784))
model.add(kr.layers.Dense(units=750, activation='relu'))
model.add(kr.layers.Dense(units=512, activation='relu'))
# Compile model - Adam optimizer for our model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Add 10 output neurons, one for each
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Read in all the files
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = f.read()
    
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = f.read()
    
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    training_images = f.read()
    
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    training_labels = f.read()

# Read all the files and save into memory
training_images = ~np.array(list(training_images[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
training_labels =  np.array(list(training_labels[8:])).astype(np.uint8)

test_images = ~np.array(list(test_images[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_labels = np.array(list(test_labels[8:])).astype(np.uint8)

# Flatten the array , 784 neurons
inputs = training_images.reshape(60000, 784)


# Encode the data into binary values
encoder = pre.LabelBinarizer()
encoder.fit(training_labels)
outputs = encoder.transform(training_labels)

