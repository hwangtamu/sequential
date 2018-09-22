import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

p = './points/'

# read csv
data_0 = np.genfromtxt(p+'128_0.csv', delimiter=',')
data_1 = np.genfromtxt(p+'128_1.csv', delimiter=',')
data_2 = np.genfromtxt(p+'128_2.csv', delimiter=',')

zero = data_0[:,[0,1,2]]
one = data_1[:,[0,1,2]]
two = data_2[:,[0,1,2]]

color = ['b', 'g', 'r']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

j=1
for (x,y,z) in combinations(range(4), 3):
    ax = fig.add_subplot(2,2,j, projection='3d')
    ax.scatter(data_0[:,x], data_0[:,y], data_0[:,z], c=color[0])
    ax.scatter(data_1[:,x], data_1[:,y], data_1[:,z], c=color[1])
    ax.scatter(data_2[:,x], data_2[:,y], data_2[:,z], c=color[2])
    j+=1
plt.show()