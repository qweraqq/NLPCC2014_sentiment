# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense, Dropout
from seya.layers.recurrent import Bidirectional
from keras.regularizers import l2

max_features = 15000
maxlen = 35  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
nb_hidden = 200
print("Loading data...")
f1 = open('training.set.txt', 'r')
f2 = open('labels.txt', 'r')
f3 = open('test.set.txt', 'r')
f4 = open('test.labels.txt', 'r')

X_training = None
y_training = None
X_test = None
y_test = None

for line in f1.readlines():
    # line = line.strip()
    # int_list = [int(s) for s in line.split()]
    # print(int_list)
    tmp = np.array(line.split())
    tmp = tmp[np.newaxis, :]

    tmp = sequence.pad_sequences(tmp, maxlen=maxlen)
    # print(tmp)
    if X_training is None:
        X_training = tmp
    else:
        X_training = np.vstack((X_training, tmp))
for line in f2.readlines():
    tmp = np.array(line.split())
    if y_training is None:
        y_training = tmp
    else:
        y_training = np.vstack((y_training, tmp))
for line in f3.readlines():
    tmp = np.array(line.split())
    tmp = tmp[np.newaxis, :]
    tmp = sequence.pad_sequences(tmp, maxlen=maxlen)
    if X_test is None:
        X_test = tmp
    else:
        X_test = np.vstack((X_test, tmp))
for line in f4.readlines():
    tmp = np.array(line.split())
    if y_test is None:
        y_test = tmp
    else:
        y_test = np.vstack((y_test, tmp))

print(X_training.shape)
print(y_training.shape)
print(X_test.shape)
print(y_test.shape)
f1.close()
f2.close()
f3.close()
f4.close()


print(len(X_training), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
print('X_train shape:', X_training.shape)
print('X_test shape:', X_test.shape)


lstm1 = LSTM(output_dim=nb_hidden/2, input_dim=nb_hidden,
             dropout_U=0.3, dropout_W=0.3, W_regularizer=l2(0.001), b_regularizer=l2(0.001))
lstm2 = LSTM(output_dim=nb_hidden/2, input_dim=nb_hidden,
             dropout_U=0.3, dropout_W=0.3, W_regularizer=l2(0.001), b_regularizer=l2(0.001))
# lstm = LSTM(output_dim=64)
# gru = LSTM(output_dim=64)
# original examples was 128, we divide by 2 because results will be concatenated
brnn = Bidirectional(forward=lstm1, backward=lstm2)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, nb_hidden, input_length=maxlen))
model.add(brnn)  # try using another Bidirectional RNN inside the Bidirectional RNN. Inception meets callback hell.
model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='RMSprop')
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=0, verbose=0, mode='auto')
print("Train...")
model.fit(X_training, y_training, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test),
          show_accuracy=True, shuffle=True, callbacks=[earlyStopping])
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
y_predict = model.predict(X_test, batch_size=1)
np.savetxt('y_predict.txt', y_predict)
print('Test score:', score)
print('Test accuracy:', acc)