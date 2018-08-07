
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
            self.splice = Splice('splice.csv')

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
                          padding=None,
                          predict=False):

        step = padding if padding>self.time_steps else self.time_steps
        hidden_states = []

        for i in range(samples):
            t = self.lstm_model.predict(np.array(x[i].reshape((-1, step, self.n_inputs))))
            hidden_states += [t]
        return hidden_states


def hidden(model, data):
    # model = load_model(model) if model else self.model
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[0].output)
    intermediate_output = intermediate_layer_model.predict(data)
    return intermediate_output


class MnistLSTMClassifier(LSTMClassifier):
    def __init__(self):
        LSTMClassifier.__init__(self,
                                time_steps=28,
                                n_units=32,
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

        self.train_model(self.x_train, self.mnist.train.labels, p="./saved_model/lstm-model_32.h5", save_model=save_model)

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if not self._data_loaded:
            self.__load_data()

        self.eval_model(self.x_test, self.mnist.test.labels)

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



class MnistLSTMClassifier2(object):
    def __init__(self):
        # Classifier
        self.time_steps = 75
        self.n_units = 128
        self.n_inputs = 4
        self.n_classes = 3
        self.batch_size = 128
        self.n_epochs = 150
        # Internal
        self._data_loaded = False
        self._trained = False
        self.__load_data()
        self.lstm_model, self.dense_model = None, None

    def __create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
        self.model.add(Dense(self.n_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def __load_data(self):
        self.splice = Splice('splice.csv')
        self._data_loaded = True

    def train(self, save_model=False):
        self.__create_model()
        if self._data_loaded == False:
            self.__load_data()

        x_train = self.splice.train['x']
        x_train = x_train.reshape((-1, x_train.shape[0], x_train.shape[1]))

        y_train = self.splice.train['y']

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
        x_test = x_test.reshape((-1, x_test.shape[0], x_test.shape[1]))

        y_test = self.splice.test['y']

        model = load_model(model) if model else self.model
        test_loss = model.evaluate(x_test, y_test)
        print(test_loss)

    def __hidden(self, model, data):
        # model = load_model(model) if model else self.model
        intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[0].output)
        intermediate_output = intermediate_layer_model.predict(data)
        return intermediate_output

    def get_hidden(self, model=None, num=0, samples=1000, padding=None, test=False, mc=False):
        """

        :param model: Keras model
        :param num: which number
        :param samples: number pf samples
        :param permute: add permutation
        :param padding: add blank padding to image
        :param test: use test data or not
        :param mc: Monte-Carlo on locating fixed points
        :return:
        """
        step = self.time_steps

        if test:
            images = self.splice.test['x']
            labels = np.argmax(self.splice.test['y'][:samples], axis=1)
        else:
            images = self.splice.train['x']
            labels = np.argmax(self.splice.train['y'][:samples], axis=1)

        # if mc:
        #     images = np.random.rand(images.shape[0], images.shape[1])

        x_test = images

        if padding:
            images = np.hstack((images, np.zeros((images.shape[0],padding-self.time_steps, 4))))
            step = padding

        x_test_padded = images

        model = load_model(model) if model else self.model
        # print(model.summary())

        x_test = x_test.reshape((-1, x_test.shape[1], x_test.shape[2]))
        x_test_padded = x_test_padded.reshape((-1, x_test_padded.shape[1], x_test_padded.shape[2]))
        predictions = model.predict_classes(x_test)
        y_test = labels
        self.lstm_model, self.dense_model = self.build_models(model)

        hidden_states = []
        results = []
        ids = []
        for i in range(samples):
            if y_test[i] == num or mc:
                hidden_states += [
                    self.__hidden(self.lstm_model, np.array(x_test_padded[i].reshape((-1, step, self.n_inputs))))]
                if y_test[i] == predictions[i] or mc:
                    results += [1]
                else:
                    results += [0]
                ids += [i]
        return hidden_states, results, ids

    def build_models(self, model=None):
        if not model:
            model = self.model

        lstm_model = Sequential()
        lstm_model.add(LSTM(self.n_units, input_shape=(None, self.n_inputs), return_sequences=True))
        weights = model.layers[0].get_weights()
        lstm_model.layers[0].set_weights(weights)

        dense_model = Sequential()
        dense_model.add(Dense(self.n_classes, input_dim=self.n_units, activation='softmax'))
        weights = model.layers[1].get_weights()
        dense_model.layers[0].set_weights(weights)

        return lstm_model, dense_model

    def real_time_predict(self, vec, num=None):
        dense_model = self.dense_model
        output = dense_model.predict(vec)
        if num == None:
            output = [max(x) for x in output]
        else:
            output = [x[num] for x in output]
        result = dense_model.predict_classes(vec)
        return result[28], list(result), output

    def vis_hidden(self, model=None, num=0, samples=100, permute=False, padding=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if self._data_loaded == False:
            self.__load_data()

        hidden_states, results, _ = self.get_hidden(model=model, num=num, samples=samples, padding=padding)

        if padding:
            self.time_steps = padding

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
            if j>0:
            #if y_test[i] == num:
                p += [i]
        for j, i in enumerate(p):
            ax = fig.add_subplot(1, len(p), j + 1)
            ax.matshow(x_test[i], cmap=cm.binary)
            ax.set_title("%s predicted as %s" % (y_test[i], predictions[i]))
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


# def softmax(x, axis=None):
#     x = x - x.max(axis=axis, keepdims=True)
#     y = np.exp(x)
#     return y / y.sum(axis=axis, keepdims=True)


if __name__ == "__main__":
    lstm_classifier = MnistLSTMClassifier()
    lstm_classifier.train(save_model=True)
    lstm_classifier.evaluate()
    # Load a trained model.
    # lstm_classifier.evaluate(model="./saved_model/lstm-model.h5")
