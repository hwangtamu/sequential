
""" gru Network.
A RNN Network (gru) implementation example using Keras.
This example is using the MNIST handwritten digits dataset (http://yann.lecun.com/exdb/mnist/)

Resources:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_gru.pdf)
    [Understanding grus](http://colah.github.io/posts/2015-08-Understanding-grus/)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
"""

# Imports
import sys, os, random

from keras.models import Sequential, Model
from keras.layers import GRU, Dense, Input
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d
from matplotlib import animation
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data
from utils import Splice, ReduceMNIST


class GRUClassifier(object):
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
        self.gru_model, self.dense_model = None, None

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
        self.model.add(GRU(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
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

        gru_model = Sequential()
        gru_model.add(GRU(self.n_units, input_shape=(None, self.n_inputs), return_sequences=True))
        weights = model.layers[0].get_weights()
        gru_model.layers[0].set_weights(weights)

        dense_model = Sequential()
        dense_model.add(Dense(self.n_classes, input_dim=self.n_units, activation='softmax'))
        weights = model.layers[-1].get_weights()
        dense_model.layers[0].set_weights(weights)
        self.gru_model, self.dense_model = gru_model, dense_model

    def eval_model(self,
                   x,
                   y,
                   model=None):
        model = load_model(model) if model else self.model
        test_loss = model.evaluate(x, y)
        print(test_loss)
        return test_loss

    def get_hidden_states(self,
                          x,
                          samples,
                          padding=None):

        step = padding if padding>self.time_steps else self.time_steps
        hidden_states = []

        for i in range(samples):
            t = self.gru_model.predict(np.array(x[i].reshape((-1, step, self.n_inputs))))
            hidden_states += [t]
        return np.array(hidden_states)


class MnistGRU(GRUClassifier):
    def __init__(self, n_units):
        GRUClassifier.__init__(self,
                                time_steps=28,
                                n_units=n_units,
                                n_inputs=28,
                                n_classes=3,
                                n_epochs=20)

    def __load_data(self):
        # self.load_data('mnist')
        #
        # x_train = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.mnist.train.images]
        # self.x_train = np.array(x_train).reshape((-1, self.time_steps, self.n_inputs))
        # self.y_train = np.array(self.mnist.train.labels)
        #
        # x_test = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.mnist.test.images]
        # self.x_test = np.array(x_test).reshape((-1, self.time_steps, self.n_inputs))
        # self.y_test = np.array(self.mnist.test.labels)
        r = ReduceMNIST()
        self.x_train = np.array(r.x_train)
        self.x_test = np.array(r.x_test)
        self.y_train = np.array(r.y_train)
        self.y_test = np.array(r.y_test)

    def train(self, save_model=False):
        if not self._data_loaded:
            self.__load_data()

        self.train_model(self.x_train,
                         self.y_train,
                         p="./saved_model/gru-mnist_"+str(self.n_units)+".h5",
                         save_model=save_model)

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if not self._data_loaded:
            self.__load_data()

        return self.eval_model(self.x_test, self.y_test, model=model)

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

        truth = np.argmax(self.y_test[:sample], axis=1) if test \
            else np.argmax(self.y_train[:sample], axis=1)

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

    def visualize(self, model=None, test=True, sample=400, padding=1000):
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

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['b', 'r', 'g']
        colors_f = ['c', 'm', 'y']
        lines = sum([ax.plot([], [], [], '-', alpha=0.5)
                     for _ in range(sample)], [])
        pts = sum([ax.plot([], [], [], 'o')
                   for _ in range(sample)], [])
        a = []
        prediction = []
        t = f = 0
        for i in range(sample):
            vec = self.dense_model.predict(hidden_states[i][0])
            s = np.argmax(vec[self.time_steps]) == truth[i]
            if s and t<15:
                a += [vec]
                prediction += [(s, truth[i])]
                t+=1
            if not s and f<15:
                a += [vec]
                prediction += [(s, truth[i])]
                f+=1
            if t>14 and f>14:
                break
            # ax.plot3D(x, y, z, c=colors[truth[i]])
            # for j in range(len(x)):
            #     ax.scatter3D(x[j], y[j], z[j], c=colors[truth[i]], alpha=j / len(x))

        def init():
            for line, pt in zip(lines, pts):
                line.set_data([], [])
                line.set_3d_properties([])

                pt.set_data([], [])
                pt.set_3d_properties([])
            return lines + pts

        def animate(i):
            i = (2 * i) % padding

            for line, pt, d in zip(lines, pts, range(len(a))):
                x, y, z = a[d][:, 0][:i+1], a[d][:, 1][:i+1], a[d][:, 2][:i+1]
                line.set_data(x, y)
                line.set_3d_properties(z)
                if prediction[d][0]:
                    line.set_color(colors[prediction[d][1]])
                else:
                    line.set_color(colors_f[prediction[d][1]])

                pt.set_data(x[-1], y[-1])
                pt.set_3d_properties(z[-1])

                if prediction[d][0]:
                    pt.set_color(colors[prediction[d][1]])
                else:
                    pt.set_color(colors_f[prediction[d][1]])

            fig.canvas.draw()
            return lines + pts

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=500, interval=30, blit=True)
        ax.view_init(45, 45)
        # plt.axis('off')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        path = './animation/gru_cc'
        if not os.path.exists(path):
            os.makedirs(path)
        anim.save(path+'/'+str(self.n_units)+'.mp4', dpi=80, writer=writer)
        # plt.show()

class DNAGRU(GRUClassifier):
    def __init__(self, n_units, n_classes, indicator):
        self.indicator = indicator
        GRUClassifier.__init__(self,
                                time_steps=60,
                                n_units=n_units,
                                n_inputs=4,
                                n_classes=n_classes,
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
                         p="./saved_model/gru-dna-mse_"+self.indicator+"_"+str(self.n_classes)+"_"+str(self.n_units)+".h5",
                         save_model=save_model)

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if not self._data_loaded:
            self.__load_data()

        return self.eval_model(self.x_test, self.y_test, model=model)

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
                _out = self.dense_model.predict(np.reshape(hidden_states[i][0][j], (-1, self.n_units)))
                #if j in output_range:
                    #stdout += [tuple([res, float('%.3f' % out)])]
                stdout+=[tuple([float('%.3f' % x) for x in _out[0]])]
                # stdout+=[tuple([float('%.3f' % x) for x in hidden_states[i][0][j]])]
                if j==self.time_steps:
                    stdout = [res] + stdout
            stdout = [self.splice.x_raw[i], truth[i]] + stdout
            print(stdout)

    def get_states(self, model=None,test=True, sample=100, padding=None):
        if not self._data_loaded:
            self.__load_data()

        x = self.x_test[:sample] if test else self.x_train[:sample]
        if padding and padding>self.time_steps:
            x = np.hstack((x,np.zeros((len(x), padding-self.time_steps, self.n_inputs))))
        truth = np.argmax(self.y_test[:sample], axis=1) if test \
            else np.argmax(self.y_train[:sample], axis=1)
        model = load_model(model) if model else self.model
        self.build_models(model)

        # build new model
        inputs1 = Input(shape=(max(padding, self.time_steps), self.n_inputs))
        gru1, state_h, state_c = GRU(self.n_units, return_sequences=True, return_state=True)(inputs1)
        gru_model = Model(inputs=inputs1, outputs=[gru1, state_h, state_c])
        weights = model.layers[0].get_weights()
        gru_model.layers[-1].set_weights(weights)

        for i in range(sample):
            hid, _, cell = gru_model.predict(np.array(x[i].reshape((-1, max(self.time_steps, padding), self.n_inputs))))
            res = [self.splice.x_raw_test[i], truth[i]]
            states = [tuple(float('%.3f' % x) for x in y) for y in hid[0]]
            a = self.make_prediction(np.reshape(hid[0][self.time_steps-1], (-1, self.n_units)))
            res+=[a[0], states[self.time_steps-1]]
            print(res)

    def dense_dots(self,model=None,test=True, sample=100, padding=None):
        if not self._data_loaded:
            self.__load_data()

        x = self.x_test[:sample] if test else self.x_train[:sample]
        if padding and padding>self.time_steps:
            x = np.hstack((x,np.zeros((len(x), padding-self.time_steps, self.n_inputs))))
        truth = np.argmax(self.y_test[:sample], axis=1) if test \
            else np.argmax(self.y_train[:sample], axis=1)
        model = load_model(model) if model else self.model
        self.build_models(model)

        # build new model
        inputs1 = Input(shape=(max(padding, self.time_steps), self.n_inputs))
        gru1, state_h, state_c = GRU(self.n_units, return_sequences=True, return_state=True)(inputs1)
        gru_model = Model(inputs=inputs1, outputs=[gru1, state_h, state_c])
        weights = model.layers[0].get_weights()
        gru_model.layers[-1].set_weights(weights)

        first = []
        second = []
        third = []

        for i in range(sample):
            hid, _, cell = gru_model.predict(np.array(x[i].reshape((-1, max(self.time_steps, padding), self.n_inputs))))
            res = [self.splice.x_raw_test[i], truth[i]]
            states = [tuple(float('%.3f' % x) for x in y) for y in hid[0]]
            for j in range(max(padding, self.time_steps)):
                a = self.make_prediction(np.reshape(hid[0][j], (-1, self.n_units)))[0]
                res+=[tuple([a, states[j]])]
            print(res)
            for j in range(2,len(res)):
                if res[j][0] == 0:
                    first += [hid[0][j-2]]
                if res[j][0] == 1:
                    second += [hid[0][j-2]]
                if res[j][0] == 2:
                    third += [hid[0][j-2]]

        np.savetxt('./points/'+str(self.n_units)+'_0.csv', np.array(first), delimiter=',')
        np.savetxt('./points/'+str(self.n_units)+'_1.csv', np.array(second), delimiter=',')
        np.savetxt('./points/'+str(self.n_units)+'_2.csv', np.array(third), delimiter=',')

    def lesion_eval(self, model=None, acuity=None):
        if not self._data_loaded:
            self.__load_data()
        if not acuity:
            acuity = 1
        model = load_model(model) if model else self.model
        self.build_models(model)

        lesion = random.sample(range(self.n_units), acuity)
        hidden_states = self.get_hidden_states(self.x_test, samples=len(self.x_test), padding=0)
        print('remove units: '+str(lesion))
        a, b, c, d = [], [], 0, 0
        for i in range(len(self.x_test)):
            res, out = self.make_prediction(np.reshape(hidden_states[i][0][self.time_steps-1], (-1, self.n_units)))
            a+=[res]
            hidden_states[i][0][:,lesion] = 0.

            res_, out_ = self.make_prediction(np.reshape(hidden_states[i][0][self.time_steps-1], (-1, self.n_units)))
            b+=[res_]

            if self.y_test[i][res]!=1 or self.y_test[i][res_]!=1:
                if self.y_test[i][res]!=1:
                    c+=1
                if self.y_test[i][res_]!=1:
                    d+=1
                print(self.splice.x_raw_test[i],np.argmax(self.y_test[i]), res, res_)
        print(c/len(self.y_test), d/len(self.y_test))

    def act_vis(self, model=None, id=0):
        if not self._data_loaded:
            self.__load_data()

        model = load_model(model) if model else self.model
        self.build_models(model)

        inputs1 = Input(shape=(self.time_steps, self.n_inputs))
        gru1, state_h = GRU(self.n_units, return_sequences=True, return_state=True)(inputs1)
        gru_model = Model(inputs=inputs1, outputs=[gru1, state_h])
        weights = model.layers[0].get_weights()
        gru_model.layers[-1].set_weights(weights)

        hidden_states = self.get_hidden_states([self.x_test[id]], samples=1, padding=0)
        hid, _ = gru_model.predict(np.array(self.x_test[id].reshape((-1, self.time_steps, self.n_inputs))))

        # print(hid.shape, cell.shape)
        plt.clf()

        title = self.splice.x_raw_test[id]
        plt.imshow(np.transpose(hidden_states[0][0]), cmap='coolwarm', interpolation='nearest',aspect='auto')
        plt.colorbar()
        plt.title(title)
        #plt.tight_layout()

        path = './figures/gru/'
        if not os.path.exists(path):
            os.makedirs(path)

        # plt.savefig(path+self.indicator+'-'+str(self.n_units)+'_'+str(id)+'.png')
        plt.show()



    def visualize(self, model=None, test=True, sample=400, padding=1000):
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

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['b', 'r', 'g']
        colors_f = ['c', 'm', 'y']
        lines = sum([ax.plot([], [], [], '-', alpha=0.5)
                     for _ in range(sample)], [])
        pts = sum([ax.plot([], [], [], 'o')
                   for _ in range(sample)], [])
        a = []
        prediction = []
        t = f = 0
        for i in range(sample):
            vec = self.dense_model.predict(hidden_states[i][0])
            s = np.argmax(vec[self.time_steps]) == truth[i]
            if s and t<15:
                a += [vec]
                prediction += [(s, truth[i])]
                t+=1
            if not s and f<15:
                a += [vec]
                prediction += [(s, truth[i])]
                f+=1
            if t>14 and f>14:
                break
            # ax.plot3D(x, y, z, c=colors[truth[i]])
            # for j in range(len(x)):
            #     ax.scatter3D(x[j], y[j], z[j], c=colors[truth[i]], alpha=j / len(x))

        def init():
            for line, pt in zip(lines, pts):
                line.set_data([], [])
                line.set_3d_properties([])

                pt.set_data([], [])
                pt.set_3d_properties([])
            return lines + pts

        def animate(i):
            i = (2 * i) % padding

            for line, pt, d in zip(lines, pts, range(len(a))):
                x, y, z = a[d][:, 0][:i+1], a[d][:, 1][:i+1], a[d][:, 2][:i+1]
                line.set_data(x, y)
                line.set_3d_properties(z)
                if prediction[d][0]:
                    line.set_color(colors[prediction[d][1]])
                else:
                    line.set_color(colors_f[prediction[d][1]])

                pt.set_data(x[-1], y[-1])
                pt.set_3d_properties(z[-1])

                if prediction[d][0]:
                    pt.set_color(colors[prediction[d][1]])
                else:
                    pt.set_color(colors_f[prediction[d][1]])

            fig.canvas.draw()
            return lines + pts

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=500, interval=30, blit=True)
        ax.view_init(45, 45)
        # plt.axis('off')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        path = './animation/'+self.indicator+'_mse'
        if not os.path.exists(path):
            os.makedirs(path)
        anim.save(path+'/'+str(self.n_units)+'.mp4', dpi=80, writer=writer)
        # plt.show()
