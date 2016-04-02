# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import jieba
import codecs
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb

from seya.layers.recurrent import Bidirectional
import six.moves.cPickle as pickle

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

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

    tmp = sequence.pad_sequences(tmp, maxlen=25,)
    print(tmp)
    if X_training is None:
        X_training = tmp
    else:
        X_training = np.vstack((X_training, tmp))
for line in f2.readlines():
    tmp = np.array(line)
    if y_training is None:
        y_training = tmp
    else:
        y_training = np.vstack((y_training, tmp))
for line in f3.readlines():
    tmp = np.array(line)
    tmp = sequence.pad_sequences(tmp, maxlen=30)
    if X_test is None:
        X_test = tmp
    else:
        X_test = np.vstack((X_test, tmp))
for line in f4.readlines():
    tmp = np.array(line)
    if y_test is None:
        y_test = tmp
    else:
        y_test = np.vstack((y_test, tmp))

print(X_training.shape())
print(y_training.shape())
print(X_test.shape())
print(y_test.shape())
f1.close()
f2.close()
f3.close()
f4.close()

# f = open("imdb.pkl", 'rb')
# train_set = pickle.load(f)
# print(train_set)
# (X_train, y_train), (X_test, y_test) = imdb.load_data(path="E:\\projects\\NLPCC2014_sentiment\\imdb.pkl",
#                                                       nb_words=max_features, test_split=0.2)
# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')

# print("Pad sequences (samples x time)")
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
#
# lstm = LSTM(output_dim=64)
# gru = GRU(output_dim=64)  # original examples was 128, we divide by 2 because results will be concatenated
# brnn = Bidirectional(forward=lstm, backward=gru)
#
# print('Build model...')
# model = Sequential()
# model.add(Embedding(max_features, 128, input_length=maxlen))
# model.add(brnn)  # try using another Bidirectional RNN inside the Bidirectional RNN. Inception meets callback hell.
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
#
# print("Train...")
# model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)
# score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
# print('Test score:', score)
# print('Test accuracy:', acc)