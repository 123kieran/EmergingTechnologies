# Imports
import keras as kr
import numpy as np
import sklearn.preprocessing as pre
import gzip



# Read in all the files
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = f.read()
    
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = f.read()
    
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    training_images = f.read()
    
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    training_labels = f.read()
