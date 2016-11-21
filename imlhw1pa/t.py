from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
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
    plt.show(block=False)

#Implement KNN
def knn(images, labels, image, k):
    images = images.astype('int')
    image = image.astype('int')
    dists = np.linalg.norm(images - image, axis=1)
    nearest = np.argsort(dists)[:k]
    return round(np.mean(labels[nearest]))

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
    res = [0] * 100
    for k in range(1, 101):
        res[k] = measure(k=k)
    return res
