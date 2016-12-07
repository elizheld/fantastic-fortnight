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
random.seed(14)

# this is the size of our encoded representations number of pca
encoding_dim = 16  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(64,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

digits = datasets.load_digits()
target_names = digits.target_names

X_digits = digits.data
y_digits = digits.target
y_digits_round = [0] * len(y_digits)
for i in range(len(y_digits)):
    if y_digits[i]>4:
        y_digits_round[i] = 1
        
n_samples, h, w = digits.images.shape          

X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits_round, test_size=0.25, random_state=42)

X_train = X_train.astype('float32') / 255.
X_test =X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
print(X_train.shape)
print(X_test.shape)
#Now let's train our autoencoder for 50 epochs:

autoencoder.fit(X_train, X_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))
                
encoded_imgs = encoder.predict(X_test)
encoded_imgs_train = encoder.predict(X_train)

grid = {
        'C': np.power(10.0, np.arange(-10, 10))
         , 'solver': ['newton-cg']
    }
clf = LogisticRegression()#penalty='l2', random_state=777, max_iter=10000, tol=10)
gs = GridSearchCV(clf, grid)
gs.fit(encoded_imgs_train, y_train)
y_pred = gs.predict(encoded_imgs)
C = confusion_matrix(y_test, y_pred, labels=range(2))
print(C)
print(np.diag(C) / map(float, np.sum(C,1)))

fpr, tpr, _ = metrics.roc_curve(np.array(y_test), gs.predict_proba(encoded_imgs)[:,1])
roc_auc = metrics.auc(fpr, tpr)

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

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i].reshape(8, 8))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(encoded_imgs[i].reshape(4, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

