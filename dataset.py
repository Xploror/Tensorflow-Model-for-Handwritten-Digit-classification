import cv2
import os 
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import tensorflow as tf

wdir = 'D:\project\images'
mnist = 'D:\mnist_784.csv'

dataset = []

# Data from mnist dataset (50000 examples)
df = pd.read_csv(mnist, header=None, nrows=50000, skiprows=1)
df.dropna(axis=1)
data = df.to_numpy(dtype='float32')
examples = data.shape[0]
feature = data[:,:-1].reshape(examples, 28, 28)
label = data[:,-1]
for example in range(examples):
    f = feature[example]
    l = label[example].astype('int')
    dataset.append([f, l])

# Data from web scraped folder in D drive (1027 examples)
for i in range(10):
    path = os.path.join(wdir, str(i))
    for img in os.listdir(path):
        array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        resize_array = cv2.resize(array, (28,28))
        dataset.append([resize_array, i])

random.shuffle(dataset)

X = []
Y = []
for features, labels in dataset:
    X.append(features)
    Y.append(labels)

#Reshaping of features(m, nx, nx, 1) and labels(m, 1)
X = np.array(X).reshape(-1, 28, 28, 1).astype('float32')
y = tf.one_hot(Y, depth=10)

#Training and test sets
X_train = X[:(len(dataset)-200),:,:,:]
X_test = X[(len(dataset)-200):,:,:,:]
y_train = y[:(len(dataset)-200),:]
y_test = y[(len(dataset)-200):,:]
