from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Data(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets("mnist", one_hot=True)

    def visualize_by_id(self, id, test=True):
        if test:
            data = self.mnist.test.images
            y = np.argmax(self.mnist.test.labels, axis=1)
        else:
            data = self.mnist.train.images
            y = np.argmax(self.mnist.train.labels, axis=1)

        x = [x.reshape((-1, 28, 28)) for x in data]
        x = np.array(x).reshape((-1, 28, 28))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(x[id], cmap=cm.binary)
        ax.set_title("%s" % (y[id]))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


