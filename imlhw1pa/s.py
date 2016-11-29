from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import pickle
import itertools

def apply_concept(lst, C):
    CP, CN = C
    r = 0
    for i in range(len(lst)):
        e = lst[i]
        if e & CP != 0: r |= (1<<i)
        if (~e) & CN != 0: r |= (1<<i)
    return r

def count_divisions(lst, H):
    s = set()
    for CP in range(2**H):
        for CN in range(2**H):
            s.add(apply_concept(lst, (CP, CN)))
    return len(s)

def lsts(H, n):
    for e in itertools.combinations(range(2**H), n):
        yield e

def find_max_divisions(H, n):
    _max = 0
    for lst in lsts(H, n):
        c = count_divisions(lst, H)
        if c > _max:
            _max = c
            #print "found example with", c, "divisions"
            if c == 2**H:
                print "SHUTTER (H:",H,"n:",n,")"
                print lst

def go(H):
    for n in range(2**H):
        find_max_divisions(H, n)
