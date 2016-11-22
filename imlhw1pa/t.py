from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

import numpy.random
try:
    idx
except NameError:
    print("creating train and test.")
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
    plt.show()

#Implement KNN
def knn(images, labels, image, k):
    images = images.astype('int')
    image = image.astype('int')
    dists = np.linalg.norm(images - image, axis=1)
    nearest = np.argsort(dists)[:k]
    return stats.mode(labels[nearest]).mode[0]

#Measure KNN
def measure(k=10, n=1000, verbose=True):
    start = time.time()
    bad = 0
    for i in range(len(test)):
        image = test[i]
        label = test_labels[i]
        predicted = knn(train[:n], train_labels[:n], image, k)
        if predicted != label:
            bad += 1
    correct_ratio = 1 - float(bad)/float(len(test))
    end = time.time()
    if verbose:
        print "k:", k, "n:", n, "correct_ratio:", correct_ratio, "elapsed:", end - start
    return correct_ratio

#Try various Ks
def try_various_ks():
    res = {}
    for k in range(1, 101):
        res[k] = measure(k=k)
    return res

#Try various Ns
def try_various_ns():
    res = {}
    for n in range(100, 5000 + 1, 100):
        res[n] = measure(k=1, n=n)
    return res

#Draw samples for question 2
def draw_samples(m=100):
    x = numpy.random.uniform(0,1,m)
    y = numpy.random.uniform(0,1,m)
    for i in range(m):
        p = 0
        if x[i] < 0.25:
            p = 0.8
        elif x[i] < 0.5:
            p = 0.1
        elif x[i] < 0.75:
            p = 0.8
        else:
            p = 0.1
        y[i] = 1 if y[i] < p else 0
    return x, y

from intervals import find_best_interval

#Plot 2_a
def plot_2a():
    for x in (0.25, 0.5, 0.75):
        plt.plot([x,x],[-.1,1.1],'k--')
    x, y = draw_samples(100)
    plt.plot(x, y, 'k.')
    plt.ylim([-.1,1.1])
    idx = numpy.argsort(x)
    intervals = find_best_interval(x[idx], y[idx], k=2)
    for interval in intervals[0]:
        print interval
        plt.plot(interval, [0.5,0.5], 'k', linewidth=10)
    plt.show()
