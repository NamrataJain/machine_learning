# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:53:41 2020

@author: Namrata
"""

# organize imports
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('E:/ML/pima-indians-diabetes-database//dataset1.txt', delimiter=',', skiprows=1)

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create the model
#model = Sequential()
#model.add(Dense(8, input_dim=8, init='uniform', activation='relu'))
#model.add(Dense(6, init='uniform', activation='relu'))
#model.add(Dense(1, init='uniform', activation='sigmoid'))


model = Sequential([Dense(8, activation='relu', input_shape=(8,)),
                    Dense(6, activation='relu'),
                    Dense(1, activation='sigmoid'),])


# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=5, verbose=0)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)
accuracy_score(Y_test, y_pred)
