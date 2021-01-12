# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:16:35 2020

@author: Namrata
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("https://raw.githubusercontent.com/mwitiderrick/kerasDO/master/HR_comma_sep.csv")

print(df.columns)

feats = ['department','salary']
df_final = pd.get_dummies(df,columns=feats,drop_first=True)
print(df_final.columns)

#print(df_final.iloc[:,-3:].head(50))
print(df_final.head(10))

X = df_final.drop(['left'],axis=1).values
y = df_final['left'].values
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(9, kernel_initializer = "uniform",activation = "relu", input_dim=18))
classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))
classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 1)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

new_pred = classifier.predict(sc.transform(np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])))

new_pred = (new_pred > 0.5)
print(new_pred)
new_pred = (new_pred > 0.6)

print(new_pred)

def make_classifier():
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = make_classifier, batch_size=10, nb_epoch=1)
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train,cv = 10)

mean = accuracies.mean()
print(mean)

variance = accuracies.var()
print(variance)


classifier1 = Sequential()
classifier1.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
classifier1.add(Dropout(rate = 0.1))
classifier1.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
classifier1.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])

new_pred1 = classifier1.predict(sc.transform(np.array([[0.26,0.7 ,3., 238., 6., 0.,0.,0.,0., 0.,0.,0.,0.,0.,1.,0., 0.,1.]])))

new_pred1 = (new_pred1 > 0.5)
print(new_pred1)
new_pred1 = (new_pred1 > 0.6)

print(new_pred1)

def make_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim=18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= optimizer,loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = make_classifier)
params = {
    'batch_size':[20,35],
    'epochs':[2,3],
    'optimizer':['adam','rmsprop']
}

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring="accuracy",
                           cv=2)

grid_search = grid_search.fit(X_train,y_train)

best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_param)
print(best_accuracy)
