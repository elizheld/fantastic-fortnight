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

# Choose the number of components for PCA 
# Previous runs have shown that 85% of the 
# variance is explained by the top 16 components
encoding_dim = 16  
N=1000
# this is our input placeholder
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


# Get simulation indices
n_train = N*len(X_train)
n_test = N*len(X_test)
ind_train = np.random.choice(np.array([0,1]), size=n_train, p=[0.67,0.33]).reshape(N, len(X_train))
ind_test =  np.random.choice(np.array([0,1]), size=n_test, p=[0.67,0.33]).reshape(N, len(X_test))
n_components = 16
    
# perform analyses for sae
hold2 = [0]*N
for i in range(N):
    x_train = X_train[ind_train.astype('bool')[i,],]
    x_test = X_test[ind_test.astype('bool')[i,],]
    y_train = Y_train[ind_train.astype('bool')[i,],]
    y_test = Y_test[ind_test.astype('bool')[i,],]
   
    
    # Build PCA
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(x_train)

    pca_digits = pca.components_.reshape((n_components, h, w))

    # Transform using PCA
    X_train_pca = pca.transform(x_train)
    X_test_pca = pca.transform(x_test)

    # Train a LR classification model
    grid = {
            'C': np.power(10.0, np.arange(-10, 10))
            , 'solver': ['newton-cg']
        }
    clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
    gs = GridSearchCV(clf, grid)
    # Fit LR
    gs.fit(X_train_pca, y_train)

    # Predict
    ypca_pred = gs.predict(X_test_pca)

    
    # Calculate FPR and TPR for ROC and AUC
    fpr, tpr, _ = metrics.roc_curve(np.array(y_test), gs.predict_proba(X_test_pca)[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    #print(roc_auc)
    print(i)
    
    hold2[i] = roc_auc
    
print(hold2)
