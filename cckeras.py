from __future__ import print_function
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations number of pca
encoding_dim = 16  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(64,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
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
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

