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

from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import Splice


class MnistLSTMClassifier(object):
    def __init__(self):
        # Classifier
        self.time_steps = 75
        self.n_units = 256
        self.n_inputs = 8
        self.n_classes = 3
        self.batch_size = 128
        self.n_epochs = 150
        # Internal
        self._data_loaded = False
        self._trained = False

    def __create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def __load_data(self):
        self.splice = Splice('splice.csv')
        self.mnist = input_data.read_data_sets("mnist", one_hot=True)
        self._data_loaded = True

    def train(self, save_model=False):
        self.__create_model()
        if self._data_loaded == False:
            self.__load_data()

        x_train = self.splice.train['x']
        y_train = self.splice.train['y']
        #x_train = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.mnist.train.images]
        #x_train = np.array(x_train).reshape((-1, self.time_steps, self.n_inputs))

        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False)

        self._trained = True

        if save_model:
            self.model.save("./saved_model/lstm-model_256.h5")

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if self._data_loaded == False:
            self.__load_data()

        x_test = self.splice.test['x']
        y_test = self.splice.test['y']
        #x_test = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.mnist.test.images]
        #x_test = np.array(x_test).reshape((-1, self.time_steps, self.n_inputs))

        model = load_model(model) if model else self.model
        test_loss = model.evaluate(x_test, y_test)
        print(test_loss)

    def __hidden(self, model, data):
        # model = load_model(model) if model else self.model
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('lstm_1').output)
        intermediate_output = intermediate_layer_model.predict(data)
        return intermediate_output

    def get_hidden(self, model=None, num=0, samples=1000, permute=False):
        if permute:
            self.mnist.test.images[0:samples, 23 * 28:28 * 28] = 0

        images = self.mnist.test.images
        x_test = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in images][:samples]
        x_test = np.array(x_test).reshape((-1, self.time_steps, self.n_inputs))
        model = load_model(model) if model else self.model
        print(model.summary())

        predictions = model.predict_classes(x_test)
        y_test = np.argmax(self.mnist.test.labels[:samples], axis=1)

        lstm_model = Sequential()
        lstm_model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True))
        weights = model.layers[0].get_weights()
        lstm_model.layers[0].set_weights(weights)

        hidden_states = []
        results = []
        for i in range(samples):
            if y_test[i] == num:
                hidden_states += [
                    self.__hidden(lstm_model, np.array(x_test[i].reshape((-1, self.time_steps, self.n_inputs))))]
                if y_test[i] == predictions[i]:
                    results += [1]
                else:
                    results += [0]
        return hidden_states, results

    def vis_hidden(self, model=None, num=0, samples=100, permute=False):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if self._data_loaded == False:
            self.__load_data()

        hidden_states, results = self.get_hidden(model, num, samples, permute)

        for j in range(self.n_units):
            for i, state in enumerate(hidden_states):
                color = 'b' if results[i] == 1 else 'r'
                state = np.transpose(state)
                plt.plot(np.array(list(range(self.time_steps))), state[j], c=color)
            plt.show()

    def visualize(self, model=None, num=0, samples=100, permute=False):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if self._data_loaded == False:
            self.__load_data()

        if permute:
            self.mnist.test.images[0:samples, 20 * 28:28 * 28] = 0

        images = self.mnist.test.images
        x_test = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in images][:samples]
        x_test = np.array(x_test).reshape((-1, self.time_steps, self.n_inputs))

        model = load_model(model) if model else self.model
        predictions = model.predict_classes(x_test)
        y_test = np.argmax(self.mnist.test.labels[:samples], axis=1)

        fig = plt.figure()
        p = []
        for i, j in enumerate(predictions - y_test):
            # if j>0:
            if y_test[i] == num:
                p += [i]
        for j, i in enumerate(p):
            ax = fig.add_subplot(1, len(p), j + 1)
            ax.matshow(x_test[i], cmap=cm.binary)
            ax.set_title("%s predicted as %s" % (y_test[i], predictions[i]))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


if __name__ == "__main__":
    lstm_classifier = MnistLSTMClassifier()
    lstm_classifier.train(save_model=True)
    lstm_classifier.evaluate()
    # Load a trained model.
    # lstm_classifier.evaluate(model="./saved_model/lstm-model.h5")
