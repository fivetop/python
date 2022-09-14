import tensorflow as tf
from time import time
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

import os, glob
import nltk
import sys

from sklearn import preprocessing

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder = preprocessing.LabelEncoder()


class TextFile(object):
    def __init__(self, filename, traintextlength, lstmsize, dropout):
        self.filename = filename
        self.traintextlength = traintextlength
        self.lstmsize = lstmsize
        self.dropout = dropout

    def tokenize(self):
        with open(self.filename, 'r') as sentencedata:
            sentence = sentencedata.read()
        sentence = str(sentence).lower()
        tokenize = nltk.word_tokenize(sentence)
        textpos = nltk.pos_tag(tokenize)

        textchars = sorted(list(set(sentence)))
        textcharinteger = dict((c, i) for i, c in enumerate(textchars))
        print(textchars)

        empty_wordlist = []

        for i in range(len(tokenize)):
            tokenize_result = tokenize[i]
            empty_wordlist.append(tokenize_result)

        wordarray = np.array(empty_wordlist)
        #print(wordarray)

        return sentence, textcharinteger, textcharinteger

    def onehotencode(self):
        origtext, textchars, integerversion = self.tokenize()
        print("Total Alphabet Size: " + str(len(textchars)))
        lstm_input = []
        lstm_output = []
        for character in range(0, len(origtext) - self.traintextlength):
            input = origtext[character : character + self.traintextlength]
            output = origtext[character + self.traintextlength]
            lstm_input.append([integerversion[characterval] for characterval in input])
            lstm_output.append(integerversion[output])
        training_data_size = len(lstm_input)
        output_data_size = len(lstm_output)

        print(training_data_size)
        print(output_data_size)

        return lstm_input, lstm_output, training_data_size

    def run_lstm(self):
        X, Y, input_datasize = self.onehotencode()
        X = np.reshape(X, (input_datasize, self.traintextlength, 1))

        X = X / float(input_datasize)
        Y = keras.utils.to_categorical(Y)
        print(X)
        print(Y)
        print("------------------Running LSTM-------------------------")

        model = Sequential()
        model.add(LSTM(self.lstmsize, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.lstmsize))
        model.add(Dropout(self.dropout))

        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        tensorboard = TensorBoard(log_dir="logs/{}", histogram_freq=0, batch_size=32, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


        base = "models/"
        filepath = base + "model.hdf5"
        print(Y)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
        model.fit(X, Y, epochs=1000, batch_size=128, callbacks=[checkpoint, tensorboard])

    def generate_text(self, modelfile):

        with open(self.filename, 'r',  encoding='utf8') as sentencedata:
            text_gensentence = sentencedata.read()
            text_gensentence = text_gensentence.lower()

        textlength = len(text_gensentence)
        chars_gensentence = sorted(list(set(text_gensentence)))
        textcharinteger = dict((c, i) for i, c in enumerate(chars_gensentence))
        textintegerchar = dict((i, c) for i, c in enumerate(chars_gensentence))

        characters = len(chars_gensentence)

        xhatorig = []
        yhat = []

        for i in range(0, textlength - self.traintextlength, 1):
            seq_in = text_gensentence[i:i + self.traintextlength]
            seq_out = text_gensentence[i + self.traintextlength]
            xhatorig.append([textcharinteger[char] for char in seq_in])
            yhat.append(textcharinteger[seq_out])
        length_number = len(xhatorig)

        xhat = np.reshape(xhatorig, (length_number, self.traintextlength, 1))
        xhat = xhat / float(characters)
        yhat = keras.utils.to_categorical(yhat)

        reload = Sequential()
        reload.add(LSTM(self.lstmsize, return_sequences=True, input_shape=(xhat.shape[1], xhat.shape[2])))
        reload.add(Dropout(self.dropout))
        reload.add(LSTM(self.lstmsize))
        reload.add(Dropout(self.dropout))
        reload.add(Dense(yhat.shape[1], activation='softmax'))

        reload.load_weights(modelfile)

        reload.compile(loss='categorical_crossentropy', optimizer='adam')

        gen = np.random.randint(0, len(xhatorig)-1)
        pattern = xhatorig[gen]
        display = ""
        # generate characters
        print(textintegerchar)
        for i in range(100):
            xhatpred = np.reshape(pattern, (1, len(pattern), 1))
            xhatpred = xhatpred / float(characters)
            prediction = reload.predict(xhatpred)
            print(prediction)
            index = np.argmax(prediction)
            print(index)
            result = textintegerchar[index]
            seq_in = [textintegerchar[value] for value in pattern]
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
            display = display + str(result)

        print(display)
