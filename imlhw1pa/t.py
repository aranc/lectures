from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

import numpy.random
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :]
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :]
test_labels = labels[idx[10000:]]

#Utility function, e.g. show(train[57])
def show(image):
    image = image.reshape(28,28)
    print(image.shape)
    plt.imshow(image)
    plt.show(block=False)

#Question 1a
def knn(images, labels, image, k):
    dists = np.square(images - image)

