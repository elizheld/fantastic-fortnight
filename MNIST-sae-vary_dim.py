#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:13:42 2017

@author: elizabethheld
"""

## Import packages and dependencies
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import metrics
import random 
import numpy as np
from keras.datasets import mnist
random.seed(14)
# Simulations
N=100 # number of simulations
encoding_dims = np.linspace(1.0, 500.0, num=10).astype(int)
hold = np.zeros((len(encoding_dims), N))
# this is our input placeholder
input_img = Input(shape=(784,))

for j in range(len(encoding_dims)):
    encoding_dim = encoding_dims[j]
    # Build the Simple autoencoder with only one hidden layer
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='relu')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    # Read in real data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # Get simulation indices
    n_train = N*len(X_train)
    n_test = N*len(X_test)
    ind_train = np.random.choice(np.array([0,1]), size=n_train, p=[0.67,0.33]).reshape(N, len(X_train))
    ind_test =  np.random.choice(np.array([0,1]), size=n_test, p=[0.67,0.33]).reshape(N, len(X_test))
        
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
        
        h = 28
        w = 28
        n_components = 64
        
        x_train = x_train.reshape((len(x_train),28*28))
        x_test = x_test.reshape((len(x_test),28*28))
        
        # More data cleaning and priming
        x_train = x_train.astype('float32') / 255.
        x_test =  x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test =  x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        #N Train our autoencoder for 100 epochs:
        if i < 10:
            autoencoder.fit(x_train, x_train,
                        nb_epoch=50,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        # else
        else:
            autoencoder.fit(x_train, x_train,
                        nb_epoch=3,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        # Encoded the images for test and training set
        encoded_imgs = encoder.predict(x_test)
        encoded_imgs_train = encoder.predict(x_train)
    
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
        hold2[i] = roc_auc
     
        
    hold[j,] = hold2
    print(j)
    print(hold)
