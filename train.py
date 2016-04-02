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
line_count = 0
min_count = 2
word_count_dict = {}
with open('sample.positive.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith('<') or len(line)<3:
            continue
        line_count += 1
        words = jieba.cut(line)
        for word in words:
            if word not in word_count_dict:
                word_count_dict[word] = 1
            else:
                word_count_dict[word] += 1
with open('sample.negative.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith('<') or len(line)<3:
            continue
        line_count += 1
        words = jieba.cut(line)
        for word in words:
            if word not in word_count_dict:
                word_count_dict[word] = 1
            else:
                word_count_dict[word] += 1

word_dict = {}
idx = 3
for word in word_count_dict:
    if word_count_dict[word] >= min_count:
        word_dict[word] = idx
        idx += 1

f = codecs.open('word_dict.txt', 'w', encoding='utf8')
for word in word_dict:
    print('%s %d' % (word, word_dict[word]), file=f)


f1 = codecs.open('training.set.txt', 'w', encoding='utf8')
f2 = codecs.open('training.labels.txt', 'w', encoding='utf8')
with open('sample.positive.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith('<') or len(line) < 3:
            continue
        words = jieba.cut(line)
        for word in words:
            if word not in word_dict:
                print('2', end=' ', file=f1)
            else:
                print(word_dict[word], end=' ', file=f1)
        print('', file=f1)
        print('1', file=f2)
with open('sample.negative.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith('<') or len(line)<3:
            continue
        words = jieba.cut(line)
        for word in words:
            if word not in word_dict:
                print('2', end=' ', file=f1)
            else:
                print(word_dict[word], end=' ', file=f1)
        print('', file=f1)
        print('0', file=f2)

f3 = codecs.open('test.set.txt', 'w', encoding='utf8')
f4 = codecs.open('test.labels.txt', 'w', encoding='utf8')
with open('test.label.cn.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith('<review id='):
            idx = line.find("label=")
            print(line[idx+7], file=f4)
        words = jieba.cut(line)
        if line.startswith("<") or len(line)<3:
            continue

        for word in words:
            if word not in word_dict:
                print('2', end=' ', file=f1)
            else:
                print(word_dict[word], end=' ', file=f3)
        print('', file=f3)

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