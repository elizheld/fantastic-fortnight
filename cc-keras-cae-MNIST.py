from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random 
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Layer
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
random.seed(14)
# apply a 3x3 convolution with 64 output filters on a 256x256 image:
#model = Sequential()
#model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=(1, 28, 28)))
# now model.output_shape == (None, 64, 256, 256)
#model.add(MaxPooling2D((2, 2), border_mode='same'))
# add a 4x4 convolution on top, with 28 output filters:

input_img = Input(shape=(1, 28, 28))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
encoded = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(28, 2, 2, activation='relu', border_mode='same')(encoded)
decoded = UpSampling2D((1, 2))(x)
#decoded = Convolution2D(1, 2, 7, activation='sigmoid', border_mode='same')(x)
encoder = Model(input=input_img, output=encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()





(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, (len(X_train), 1, 28, 28))
X_test = np.reshape(X_test, (len(X_test), 1, 28, 28))
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

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

# More data cleaning and priming
#N Train our autoencoder for 100 epochs:
autoencoder.fit(X_train, X_train,
                nb_epoch=1,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

#X_train = X_train.reshape((60000,28*28))
#X_test = X_test.reshape((10000,28*28))

# Encoded the images for test and training set
encoded_imgs = encoder.predict(X_train)
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
plt.title('Receiver operating characteristic After CAE')
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
        plt.imshow(encoded_imgs[1].reshape((28, 28)), cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
model.layers[-1]
# Plot the images
#plot_gallery(X_test, h, w)
#plot_gallery(encoded_imgs, 8, 8)

#plt.show()
