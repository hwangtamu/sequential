
""" LSTM Network.
A RNN Network (LSTM) implementation example using Keras.
This example is using the MNIST handwritten digits dataset (http://yann.lecun.com/exdb/mnist/)

Resources:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
"""

# Imports
import sys

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data
from utils import Splice


class LSTMClassifier(object):
    def __init__(self,
                 time_steps,
                 n_units,
                 n_inputs,
                 n_classes,
                 batch_size = 128,
                 n_epochs = 150):
        self.time_steps = time_steps
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model = None

        # Internal
        self._data_loaded = False
        self._trained = False
        self.lstm_model, self.dense_model = None, None

    def load_data(self, task):
        if task=='mnist':
            self.mnist = input_data.read_data_sets("mnist", one_hot=True)
        if task=='dna':
            self.splice = Splice('splice.csv', n_class=self.n_classes)

        self._data_loaded=True

    def create_model(self,
                     loss='categorical_crossentropy',
                     metric='accuracy'):
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss=loss,
                           optimizer='rmsprop',
                           metrics=[metric])

    def train_model(self,
              x,
              y,
              p,
              save_model=False):
        self.create_model()
        self.model.fit(x, y, batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False)
        self._trained = True

        if save_model:
            self.model.save(p)

    def build_models(self, model=None):
        if not model:
            model = self.model

        lstm_model = Sequential()
        lstm_model.add(LSTM(self.n_units, input_shape=(None, self.n_inputs), return_sequences=True))
        weights = model.layers[0].get_weights()
        lstm_model.layers[0].set_weights(weights)

        dense_model = Sequential()
        dense_model.add(Dense(self.n_classes, input_dim=self.n_units, activation='softmax'))
        weights = model.layers[-1].get_weights()
        dense_model.layers[0].set_weights(weights)
        self.lstm_model, self.dense_model = lstm_model, dense_model

    def eval_model(self,
                   x,
                   y,
                   model=None):
        model = load_model(model) if model else self.model
        test_loss = model.evaluate(x, y)
        print(test_loss)

    def get_hidden_states(self,
                          x,
                          samples,
                          padding=None):

        step = padding if padding>self.time_steps else self.time_steps
        hidden_states = []

        for i in range(samples):
            t = self.lstm_model.predict(np.array(x[i].reshape((-1, step, self.n_inputs))))
            hidden_states += [t]
        return hidden_states


class MnistLSTM(LSTMClassifier):
    def __init__(self, n_units):
        LSTMClassifier.__init__(self,
                                time_steps=28,
                                n_units=n_units,
                                n_inputs=28,
                                n_classes=10,
                                n_epochs=20)

    def __load_data(self):
        self.load_data('mnist')

        x_train = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.mnist.train.images]
        self.x_train = np.array(x_train).reshape((-1, self.time_steps, self.n_inputs))

        x_test = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.mnist.test.images]
        self.x_test = np.array(x_test).reshape((-1, self.time_steps, self.n_inputs))

    def train(self, save_model=False):
        if not self._data_loaded:
            self.__load_data()

        self.train_model(self.x_train,
                         self.mnist.train.labels,
                         p="./saved_model/lstm-mnist"+str(self.n_units)+".h5",
                         save_model=save_model)

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if not self._data_loaded:
            self.__load_data()

        self.eval_model(self.x_test, self.mnist.test.labels, model=model)

    def make_prediction(self, vec, num=None):
        output = self.dense_model.predict(vec)
        if num == None:
            output = [max(x) for x in output]
        else:
            output = [x[num] for x in output]
        result = self.dense_model.predict_classes(vec)
        return result[0], output[0]

    def real_time_predict(self, model=None, test=True, sample=1000, padding=None):
        if not self._data_loaded:
            self.__load_data()

        x = self.x_test[:sample] if test else self.x_train[:sample]

        if padding and padding>self.time_steps:
            x = np.hstack((x,np.zeros((len(x), padding-self.time_steps, self.n_inputs))))

        truth = np.argmax(self.mnist.test.labels[:sample], axis=1) if test \
            else np.argmax(self.mnist.train.labels[:sample], axis=1)

        model = load_model(model) if model else self.model
        self.build_models(model)

        hidden_states = self.get_hidden_states(x, samples=sample, padding=padding)

        for i in range(sample):
            stdout = [truth[i]]
            for j in range(max(padding,self.time_steps)):
                res, out = self.make_prediction(np.reshape(hidden_states[i][0][j], (-1, self.n_units)))
                if j%4==0:
                    stdout += [tuple([res, float('%.3f' % out)])]
            print(stdout)


class DNALSTM(LSTMClassifier):
    def __init__(self, n_units):
        LSTMClassifier.__init__(self,
                                time_steps=60,
                                n_units=n_units,
                                n_inputs=4,
                                n_classes=2,
                                n_epochs=100)

    def __load_data(self):
        self.load_data('dna')
        self.x_train = self.splice.train['x']
        self.x_test = self.splice.test['x']
        self.y_train = self.splice.train['y']
        self.y_test = self.splice.test['y']

    def train(self, save_model=False):
        if not self._data_loaded:
            self.__load_data()

        self.train_model(self.x_train,
                         self.y_train,
                         p="./saved_model/lstm-dna_"+str(self.n_units)+".h5",
                         save_model=save_model)

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if not self._data_loaded:
            self.__load_data()

        self.eval_model(self.x_test, self.y_test, model=model)

    def make_prediction(self, vec, num=None):
        output = self.dense_model.predict(vec)
        if num == None:
            output = [max(x) for x in output]
        else:
            output = [x[num] for x in output]
        result = self.dense_model.predict_classes(vec)
        return result[0], output[0]

    def real_time_predict(self, model=None, test=True, sample=100, padding=None):
        if not self._data_loaded:
            self.__load_data()

        x = self.x_test[:sample] if test else self.x_train[:sample]

        if padding and padding>self.time_steps:
            x = np.hstack((x,np.zeros((len(x), padding-self.time_steps, self.n_inputs))))
        truth = np.argmax(self.y_test[:sample], axis=1) if test \
            else np.argmax(self.y_train[:sample], axis=1)
        model = load_model(model) if model else self.model
        self.build_models(model)

        hidden_states = self.get_hidden_states(x, samples=sample, padding=padding)

        for i in range(sample):
            stdout = []
            for j in range(max(padding,self.time_steps)):
                res, out = self.make_prediction(np.reshape(hidden_states[i][0][j], (-1, self.n_units)))
                #if j%1==0:
                stdout += [tuple([res, float('%.3f' % out)])]
                if j==self.time_steps:
                    stdout = [res] + stdout
            stdout = [self.splice.x_raw[i], truth[i]] + stdout
            print(stdout)
