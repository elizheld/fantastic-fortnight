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

# Choose the number of components for PCA 
# Previous runs have shown that 85% of the 
# variance is explained by the top 16 components
encoding_dim = 64  

# this is our input placeholder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)

autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.array(X_train)
X_test = np.array(X_test)

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
n_components = 64
print(np.shape(X_train), np.shape(X_test))
X_train = X_train.reshape((60000,28*28))
X_test = X_test.reshape((10000,28*28))

# More data cleaning and priming
X_train = X_train.astype('float32') / 255.
X_test =X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

#N Train our autoencoder for 100 epochs:
autoencoder.fit(X_train, X_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test))

# Encoded the images for test and training set
encoded_imgs = encoder.predict(X_test)
encoded_imgs_train = encoder.predict(X_train)

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
C = confusion_matrix(y_test, y_pred, labels=range(2))
print(C)
print(np.diag(C) / map(float, np.sum(C,1)))

# Calculate FPR and TPR for ROC and AUC
fpr, tpr, _ = metrics.roc_curve(np.array(y_test), gs.predict_proba(encoded_imgs)[:,1])
roc_auc = metrics.auc(fpr, tpr)
print(roc_auc)

# Plot ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic After AE')
plt.legend(loc="lower right")
plt.show()

# Plot an example of what the digits look like before and 
# After dimension reduction
# This Function was based off block of code from another coder online
def plot_gallery(images, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

# Plot the images
plot_gallery(X_test, h, w)
plot_gallery(encoded_imgs, 8, 8)

plt.show()
