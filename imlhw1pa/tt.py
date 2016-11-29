from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import pickle

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

print "Measure k=10 n=1000"
measure()
