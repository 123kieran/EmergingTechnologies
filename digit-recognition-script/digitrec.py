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

#Build neural network 
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

print("-------------------Welcome---------------------------------")
print("Would you like to train a dataset? Or load the data you have?")
print("Enter Y to train data set")
print("Enter N to load your own data")
option = input("y/n : ")
if option == 'y':
    #Train - This will train the model.
    model.fit(inputs, outputs, epochs=10, batch_size=100)
    
    # Save the model to local disk
    model.save("data/model.h5")

    from random import randint
    
    for i in range(10): #Run 20 tests
        print("----------------------------------")
        randIndex = randint(0, 9999) #Get a random index to pull an image from
        test = model.predict(test_images[randIndex:randIndex+1]) #Pull the image from the dataset
        result = test.argmax(axis=1) #Set result to the highest array value
        print("The actual number:  ", test_labels[randIndex:randIndex+1])
        print("The network reads:  ", result)
        print("----------------------------------")
        
    #print out accuracy
    metrics = model.evaluate(inputs, outputs, verbose=0)
    print("Metrics(Test loss & Test Accuracy): ")
    print(metrics)

    # Evaluates and then prints error rate accuracy
    scores = model.evaluate(inputs, outputs, verbose=2)
    print("Error Rate: %.2f%%" % (100-scores[1]*100))

    print((encoder.inverse_transform(model.predict(test_images)) == test_labels).sum())

elif option == 'n':
    model.load_weights("data/model.h5")
else:
    print("*Invalid option selected*")
    print("**Model will not produce accurate predictions due to lack of training**")

