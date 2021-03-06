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

digits = datasets.load_digits()
target_names = digits.target_names

X_digits = digits.data
y_digits = digits.target
y_digits_round = [0] * len(y_digits)
for i in range(len(y_digits)):
    if y_digits[i]>4:
        y_digits_round[i] = 1
        
n_samples, h, w = digits.images.shape    
print(h)
print(w)

X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits_round, test_size=0.25, random_state=42)

X_train = X_train.astype('float32') / 255.
X_test =X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
input_img = Input(shape=(64,))
encoded = Dense(32, activation='sigmoid')(input_img)
encoded = Dense(16, activation='sigmoid')(encoded)

decoded = Dense(16, activation='sigmoid')(encoded)
decoded = Dense(32, activation='sigmoid')(decoded)
decoded = Dense(64, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(X_train, X_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test))
#decoded_imgs = autoencoder.predict(x_test)
encoder = Model(input=input_img, output=encoded)
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

def plot_gallery(images, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

plot_gallery(X_test, h, w)
plot_gallery(encoded_imgs, 4, 4)

plt.show()
