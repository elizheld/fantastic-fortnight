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

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.array(X_train)
X_test = np.array(X_test)

# Get simulation indices
n_train = N*len(X_train)
n_test = N*len(X_test)
ind_train = np.random.choice(np.array([0,1]), size=n_train, p=[0.67,0.33]).reshape(N, len(X_train))
ind_test =  np.random.choice(np.array([0,1]), size=n_test, p=[0.67,0.33]).reshape(N, len(X_test))

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

    # Train a LR classification model
    grid = {
            'C': np.power(10.0, np.arange(-10, 10))
            , 'solver': ['newton-cg']
        }
    clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
    gs = GridSearchCV(clf, grid)
    # Fit LR
    gs.fit(x_train, y_train)

    # Predict
    ypca_pred = gs.predict(x_test)

    
    # Calculate FPR and TPR for ROC and AUC
    fpr, tpr, _ = metrics.roc_curve(np.array(y_test), gs.predict_proba(x_test)[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    #print(roc_auc)
    print(i)
    
    hold2[i] = roc_auc
    
print(hold2)
