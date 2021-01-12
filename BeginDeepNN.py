# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:35:58 2020

@author: Namrata
"""

import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#plt.imshow(x_train[8], cmap=plt.cm.binary)
print(y_train[8]) #1
print(x_train.ndim) #3
print(x_train.shape) #(60000, 28, 28)
print(x_train.dtype) #uint8

#Data normalization in Keras (the matrix of 28×28 numbers can be represented by a vector (array) of 784 numbers (concatenating row by row)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

#After executing these Python instructions, we can verify that x_train.shape takes the form of (60000, 784) and x_test.shape takes the form of (10000, 784), where the first dimension indexes the image and the second indexes the pixel in each image
print(x_train.shape) #(60000, 784)
print(x_test.shape) #(10000, 784)

print(y_test[0]) #7
print(y_train[0]) #5
print(y_train.shape) #(60000,)
print(x_test.shape) #(10000, 784)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(y_test[0]) #[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
print(y_train[0]) #[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
print(y_train.shape) #(60000, 10)
print(y_test.shape) #(10000, 10)

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

#to check the architecture of our model
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

#Generate predictions
#To test this method we can choose any element. For ease, let’s take one from the test dataset x_test. For example let’s choose the element 11 of this dataset x_test.

#plt.imshow(x_test[11], cmap=plt.cm.binary)
image =  x_test[11].shape
#plt.imshow(image)
predictions = model.predict(x_test)

np.argmax(predictions[11])
print(predictions[11])

