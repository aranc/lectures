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

#Calculate the true error for a set of intervals
def calculate_true_error(intervals):
    #calc_area integrates from _from to _to
    #the function 1 if x in intervals, 0 otherwise
    def calc_area(intervals, _from, _to):
        area = 0
        for interval in intervals:
            interval_start = interval[0]
            interval_end = interval[1]
            start = max(_from, interval_start)
            end = min(_to, interval_end)
            area += max(0, end - start)
        #print "from:",_from,"to:",_to,"area:",area
        return area
    return 0.2 * (calc_area(intervals, 0.0, 0.25)) +            \
           0.8 * (.25 - (calc_area(intervals, 0.0, 0.25))) +    \
           0.9 * (calc_area(intervals, 0.25, 0.5)) +            \
           0.1 * (.25 - (calc_area(intervals, 0.25, 0.5))) +    \
           0.2 * (calc_area(intervals, 0.5, 0.75)) +            \
           0.8 * (.25 - (calc_area(intervals, 0.5, 0.75))) +    \
           0.9 * (calc_area(intervals, 0.75, 1.0)) +            \
           0.1 * (.25 - (calc_area(intervals, 0.75, 1.0)))

#measure intervals ERM errors
def measure_intervals(m=50, k=2):
    #draw sample
    x, y = draw_samples(m)
    #sort sample
    idx = numpy.argsort(x)
    x = x[idx]
    y = y[idx]
    #run ERM
    intervals, error = find_best_interval(x, y, k)
    empirical_error = float(error) / float(m)
    true_error = calculate_true_error(intervals)
    return true_error, empirical_error

def measure_intervals_T_times(m=50, k=2, T=100):
    empirical_errors = numpy.zeros(T)
    true_errors = numpy.zeros(T)
    for t in range(T):
        true_errors[t], empirical_errors[t] = measure_intervals(m, k)
    return np.mean(true_errors), np.mean(empirical_errors)

def prepare_2c():
    empirical_errors = {}
    true_errors = {}
    k = 2
    for m in range(10, 100 + 1, 5):
        start = time.time()
        true_errors[m], empirical_errors[m] = measure_intervals_T_times(m, k, 100)
        end = time.time()
        print "m:",m, "true_error:",true_errors[m], "empirical_error:",empirical_errors[m],"elapsed:",end-start
    pickle.dump([true_errors, empirical_errors], open("2c.pkl", "w"))
    return true_errors, empirical_errors

def plot_2c():
    true_errors, empirical_errors = pickle.load(open("2c.pkl"))
    ms = range(10, 100 + 1, 5)
    plt.plot(ms, [true_errors[m] for m in ms], 'k-', label='True error')
    plt.plot(ms, [empirical_errors[m] for m in ms],'k--', label='Empirical error')
    plt.legend()
    plt.show()

def prepare_2e():
    empirical_errors = {}
    true_errors = {}
    m = 50
    for k in range(1, 20 + 1, 1):
        start = time.time()
        true_errors[k], empirical_errors[k] = measure_intervals_T_times(m, k, 100)
        end = time.time()
        print "k:",k, "true_error:",true_errors[k], "empirical_error:",empirical_errors[k],"elapsed:",end-start
    pickle.dump([true_errors, empirical_errors], open("2e.pkl", "w"))
    return true_errors, empirical_errors

def plot_2e():
    true_errors, empirical_errors = pickle.load(open("2e.pkl"))
    ks = range(1, 20 + 1, 1)
    plt.plot(ks, [true_errors[k] for k in ks], 'ko-', label='True error')
    plt.plot(ks, [empirical_errors[k] for k in ks],'ko--', label='Empirical error')
    plt.legend()
    plt.show()

#prepare_2e()
#plot_2e()

def answer_2d():
    res = {}
    ks = range(1, 20 + 1, 1)
    for k in ks:
        res[k] = measure_intervals(50, k)
        print k, res[k]
