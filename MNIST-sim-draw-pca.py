#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:53:53 2017

@author: elizabethheld
"""
# Load packages
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import random
random.seed(14)
import scipy.io

# Simulations
N=1000 # number of simulations

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
hold = [0]*N
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
    n_components = 59
    
    x_train = x_train.reshape((len(x_train),28*28))
    x_test = x_test.reshape((len(x_test),28*28))
    
    # Build PCA
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(x_train)
    pca_digits = pca.components_.reshape((n_components, h, w))

    # Transform using PCA
    x_train_pca = pca.transform(x_train)
    #print(np.shape(x_train_pca))
    x_test_pca = pca.transform(x_test)
    #print(np.shape(x_test_pca))

    # Train a LR classification model
    grid = {
            'C': np.power(10.0, np.arange(-10, 10))
             , 'solver': ['newton-cg']
        }
    clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
    gs = GridSearchCV(clf, grid)
    gn = GridSearchCV(clf, grid)
    # Fit LR
    gs.fit(x_train_pca, y_train)
    
    # Predict
    ypca_pred = gs.predict(x_test_pca)
    
    # Summarize results using confusion matrix
    #C = confusion_matrix(y_test, ypca_pred, labels=range(2))
    #print(np.diag(C) / map(float, np.sum(C,1)))
    
    # Get FPR and TPR for ROC curve and AUC
    fpr, tpr, _ = metrics.roc_curve(np.array(y_test), gs.predict_proba(x_test_pca)[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)
    print(i)
    hold[i] = roc_auc
  
print(hold)
