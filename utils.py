from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import csv, random
from collections import Counter


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

    def serial_viz_by_id(self, ids, param=None, test=True):
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


class Splice:
    def __init__(self,
                 p, n_class=2):
        self.base = {'A':0,'T':1,'G':2,'C':3,'N':4,'D':5,'R':6,'S':7}
        self.result = {'EI':0,'IE':1,'N':2}
        self.x = None
        self.y = None
        self.path = p
        self.count = None
        self.train = {}
        self.test = {}
        self.split()
        self.n_class = n_class

    def convert(self, n_class=2):
        """
        convert DNA sequences to numpy arrays
        :param p: csv file path
        :param n_class: number of classes
        :return:
        """
        with open(self.path) as csvfile:
            reader = csv.reader(csvfile)
            d = []
            for row in reader:
                d+=[row]
            if n_class==2:
                dd = []
                for i in range(len(d)):
                    if d[i][0] in ['EI','IE']:
                        dd+=[d[i]]
                d = dd

            random.shuffle(d)

            self.x = np.zeros((len(d),len(d[0][2].strip()),4))
            self.y = np.zeros((len(d),n_class))
            self.count = Counter([x[0] for x in d])
            for i in range(len(d)):
                tmp = [self.base[x] for x in d[i][2].strip()]
                for j in range(len(tmp)):
                    if tmp[j]==4:
                        # N: A or G or C or T
                        self.x[i][j][0] = .25
                        self.x[i][j][1] = .25
                        self.x[i][j][2] = .25
                        self.x[i][j][3] = .25
                    elif tmp[j]==5:
                        # D: A or G or T
                        self.x[i][j][0] = .33
                        self.x[i][j][1] = .33
                        self.x[i][j][2] = .33
                    elif tmp[j]==6:
                        # R: A or G
                        self.x[i][j][0] = .50
                        self.x[i][j][2] = .50
                    elif tmp[j]==7:
                        # S: C or G
                        self.x[i][j][2] = .50
                        self.x[i][j][3] = .50
                    else:
                        self.x[i][j][tmp[j]] = 1

                #self.x[i][range(len(tmp)),tmp] = 1
                self.y[i][self.result[d[i][0]]] = 1

    def split(self):
        self.convert()
        self.train['x'], self.test['x'] = self.x[:int(self.x.shape[0] * 0.8)], self.x[int(self.x.shape[0] * 0.8):]
        self.train['y'], self.test['y'] = self.y[:int(self.y.shape[0] * 0.8)], self.y[int(self.y.shape[0] * 0.8):]



if __name__=="__main__":
    s = Splice('splice.csv')
    print(s.train['x'].shape, s.test['y'].shape)



