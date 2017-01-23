#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:41:03 2017

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
from sklearn import metrics
import random 
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
random.seed(14)

# Choose the number of components for PCA 
# Previous runs have shown that 85% of the 
# variance is explained by the top 16 components
encoding_dim = 16  
N=1000
# this is our input placeholder
input_img = Input(shape=(64,))
encoded = Dense(32, activation='sigmoid')(input_img)
encoded = Dense(16, activation='sigmoid')(encoded)

decoded = Dense(16, activation='sigmoid')(encoded)
decoded = Dense(32, activation='sigmoid')(decoded)
decoded = Dense(64, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Load toy data
digits = datasets.load_digits()
target_names = digits.target_names
X_digits = digits.data
y_digits = digits.target

y_digits_round = y_digits > 4
y_digits_round = y_digits_round.astype('int')
# Get total number of images and their dims recorded
n_samples, h, w = digits.images.shape          

# Split the data into testing and training data
X_train, X_test, Y_train, Y_test = train_test_split(X_digits, y_digits_round, test_size=0.25, random_state=42)
X_train = X_train.astype('float32') / 255.
X_test =X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# Get simulation indices
n_train = N*len(X_train)
n_test = N*len(X_test)
ind_train = np.random.choice(np.array([0,1]), size=n_train, p=[0.67,0.33]).reshape(N, len(X_train))
ind_test =  np.random.choice(np.array([0,1]), size=n_test, p=[0.67,0.33]).reshape(N, len(X_test))

# perform analyses for sae
hold2 = [0]*N
for i in range(N):
    x_train = X_train[ind_train.astype('bool')[i,],]
    x_test = X_test[ind_test.astype('bool')[i,],]
    y_train = Y_train[ind_train.astype('bool')[i,],]
    y_test = Y_test[ind_test.astype('bool')[i,],]
    
    n_components = 16
    
    x_train = x_train.reshape((len(x_train),8*8))
    x_test = x_test.reshape((len(x_test),8*8))
    
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
    encoder = Model(input=input_img, output=encoded)
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
    print(i)
    
    hold2[i] = roc_auc
    
print(hold2)
