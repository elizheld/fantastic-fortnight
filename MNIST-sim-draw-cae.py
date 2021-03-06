#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:38:02 2017
@author: elizabethheld
"""

## Import packages and dependencies
from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random 
from keras.datasets import mnist

random.seed(14)
# Simulations
N=1000 # number of simulations

input_img = Input(shape=(1, 28, 28))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
encoded = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)


x = Convolution2D(28, 2, 2, activation='relu', border_mode='same')(encoded)
decoded = UpSampling2D((1, 2))(x)

encoder = Model(input=input_img, output=encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Read in real data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.array(X_train)
X_test = np.array(X_test)

# Get simulation indices
n_train = N*len(X_train)
n_test = N*len(X_test)
ind_train = np.random.choice(np.array([0,1]), size=n_train, p=[0.9,0.1]).reshape(N, len(X_train))
ind_test =  np.random.choice(np.array([0,1]), size=n_test, p=[0.9,0.1]).reshape(N, len(X_test))
    
# perform analyses for PCA 
hold2 = [0]*N
for i in range(N):
    x_train = X_train[ind_train.astype('bool')[i,],]
    x_test = X_test[ind_test.astype('bool')[i,],]
    y_train = Y_train[ind_train.astype('bool')[i,],]
    y_test = Y_test[ind_test.astype('bool')[i,],]
    
    y_train = y_train > 4
    y_train = y_train.astype('int')
    y_test = y_test > 4 
    y_test = y_test.astype('int')
    
   
    x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
    x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # More data cleaning and priming
    #N Train our autoencoder for 100 epochs:
    if i < 10:
        history=autoencoder.fit(x_train, x_train,
                    nb_epoch=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    # else
    else:
        history=autoencoder.fit(x_train, x_train,
                    nb_epoch=3,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    

    # Encoded the images for test and training set
    encoded_imgs = encoder.predict(x_test)
    encoded_imgs_train = encoder.predict(x_train)

    encoded_imgs = encoded_imgs.reshape((len(x_test),-1,112))
    encoded_imgs = encoded_imgs.reshape((-1,112))
    encoded_imgs_train = encoded_imgs_train.reshape((len(x_train),-1,112))
    encoded_imgs_train = encoded_imgs_train.reshape((-1,112))

    # Construct a search grid for Logistic Regression Optimization
    grid = {
            'C': np.power(10.0, np.arange(-10, 10))
             , 'solver': ['newton-cg']
        }
    
    # Logistic Regression
    clf = LogisticRegression(penalty='l2', random_state=42, max_iter=10000, tol=10)
    gs = GridSearchCV(clf, grid)
    
    # Fit the LR Model to the training data
    gs.fit(encoded_imgs_train, y_train)
    # Predict 0,1 on encoded test data
    y_pred = gs.predict(encoded_imgs)
    # Compute confusion matrix to summarize results
    #C = confusion_matrix(y_test, y_pred, labels=range(2))
    #print(C)
    #print(np.diag(C) / map(float, np.sum(C,1)))
    
    # Calculate FPR and TPR for ROC and AUC
    fpr, tpr, _ = metrics.roc_curve(np.array(y_test), gs.predict_proba(encoded_imgs)[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    #print(roc_auc)
    print(i)
    
    hold2[i] = roc_auc
    
np.savetxt('cae-sim-hold.txt', hold2, delimiter=',')
print(hold2)
