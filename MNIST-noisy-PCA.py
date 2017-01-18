@author: elizabethheld
"""
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

def sp_noise(image, prob):
    noise = np.random.rand(784) > prob
    noise = noise.astype(int)
    for i in range(len(noise)):
        if noise[i]==1:
            noise[i]=255
    output = noise + image
    return output

    

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.array(X_train)
X_test = np.array(X_test)

noise_X_train = np.zeros((60000,784))
for i in range(len(X_train)):
    noise_X_train[i] = sp_noise(X_train[i].reshape(784),0.95)
    
noise_X_test = np.zeros((10000,784))
for i in range(len(X_test)):
    noise_X_test[i] = sp_noise(X_test[i].reshape(784),0.95)

# Change y_train, y_test so that digits 0-4 are 0 
# and those 5-9 are 1
y_train_round = [0] * len(y_train)
y_test_round = [0] * len(y_test)
for i in range(len(y_train)):
    if y_train[i]>4:
        y_train_round[i] = 1
        
for i in range(len(y_test)):
    if y_test[i]>4:
        y_test_round[i] = 1
        
y_test = y_test_round
y_train = y_train_round

h = 28
w = 28
n_components = 59
print(np.shape(X_train), np.shape(X_test))
X_train = X_train.reshape((60000,28*28))
X_test = X_test.reshape((10000,28*28))
# Build PCA
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
pca_digits = pca.components_.reshape((n_components, h, w))

# Transform using PCA
X_train_pca = pca.transform(X_train)
print(np.shape(X_train_pca))
X_test_pca = pca.transform(X_test)
print(np.shape(X_test_pca))

#nsamples, nx, ny = X_train_pca.shape
#_train_pca = X_train_pca.reshape((nsamples,nx*ny))

#nsamples_test, nx_test, ny_test = X_test_pca.shape
#X_test_pca = X_test_pca.reshape((nsamples_test,nx_test*ny_test))
# Train a LR classification model
grid = {
        'C': np.power(10.0, np.arange(-10, 10))
         , 'solver': ['newton-cg']
    }
clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
gs = GridSearchCV(clf, grid)
gn = GridSearchCV(clf, grid)
# Fit LR
gs.fit(X_train_pca, y_train)
#gn.fit(X_train, y_train)

# Predict
ypca_pred = gs.predict(X_test_pca)
#y_pred = gn.predict(X_test)

# Summarize results using confusion matrix
C = confusion_matrix(y_test, ypca_pred, labels=range(2))
#Cn = confusion_matrix(y_test, y_pred, labels=range(2))
print(C)
print(np.diag(C) / map(float, np.sum(C,1)))
# Get FPR and TPR for ROC curve and AUC
fpr, tpr, _ = metrics.roc_curve(np.array(y_test), gs.predict_proba(X_test_pca)[:,1])
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic after PCA')
plt.legend(loc="lower right")
plt.show()
