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

input_img = Input(shape=(1, 8, 8))

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
##X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
##X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

X_train = X_train.reshape((1, 1, X_train.shape[0], X_train.shape[1]))
X_test = X_test.reshape((1, 1, X_test.shape[0], X_test.shape[1]))
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

nb_filters=32
nb_pool=2
nb_conv=3
d = Dense(16)
c = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', input_shape=(1, 8, 8))
mp =MaxPooling2D(pool_size=(nb_pool, nb_pool))
 # =========      ENCODER     ========================
model.add(c)
model.add(Activation('tanh'))
model.add(mp)
model.add(Dropout(0.25))
# =========      BOTTLENECK     ======================
model.add(Flatten())
model.add(d)
model.add(Activation('tanh'))
 # =========      BOTTLENECK^-1   =====================
model.add(DependentDense(nb_filters * 14 * 14, d))  
model.add(Activation('tanh'))
model.add(Reshape((nb_filters, 14, 14)))
# =========      DECODER     =========================
model.add(DePool2D(mp, size=(nb_pool, nb_pool)))
model.add(Deconvolution2D(c, border_mode='same'))
model.add(Activation('tanh'))
model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy')
model.fit(X_train, X_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test))
#decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = model.predict(X_test)
encoded_imgs_train = model.predict(X_train)

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
