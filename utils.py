from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Data(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets("mnist", one_hot=True)
        data = self.mnist.train.images
        x = [x.reshape((-1, 28, 28)) for x in data]
        self.x_train = np.array(x).reshape((-1, 28, 28))
        self.y_train = np.argmax(self.mnist.train.labels, axis=1)

        data = self.mnist.test.images
        x = [x.reshape((-1, 28, 28)) for x in data]
        self.x_test = np.array(x).reshape((-1, 28, 28))
        self.y_test = np.argmax(self.mnist.test.labels, axis=1)

    def visualize_by_id(self, id, test=True):
        if test:
            x,y = self.x_test, self.y_test
        else:
            x,y = self.x_train, self.y_train

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(x[id], cmap=cm.binary)
        ax.set_title("%s" % (y[id]))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def serial_viz_by_id(self, ids, param=None ,test=True):
        l = len(ids)
        a = int((l**0.5)*0.7)
        b = l//a+1

        if test:
            x,y = self.x_test, self.y_test
        else:
            x,y = self.x_train, self.y_train

        fig = plt.figure()
        for i in range(l):
            title = str(ids[i])
            if param:
                title += ' '+str(param[i])[:6]
            ax = fig.add_subplot(a, b, i+1)
            ax.matshow(x[ids[i]], cmap=cm.binary)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()





