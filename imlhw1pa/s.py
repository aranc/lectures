from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import pickle
import itertools

#lst: list of numbers
#C: pair of numbers
def apply_concept(lst, C):
    CP, CN = C
    r = 0
    for i in range(len(lst)):
        e = lst[i]
        if e & CP != 0: r |= (1<<i)
        if (~e) & CN != 0: r |= (1<<i)
    return r

#lst: list of numbers
def count_divisions(lst, nbits):
    s = set()
    S = set()
    for CP in range(2**nbits):
        for CN in range(2**nbits):
            r = apply_concept(lst, (CP, CN))
            if r not in s:
                s.add(r)
                S.add((r,(CP,CN)))
    return len(s), sorted(list(S))

def lsts(nbits, n):
    for e in itertools.combinations(range(2**nbits), n):
        yield e

def find_max_divisions(nbits, n):
    _max = 0
    for lst in lsts(nbits, n):
        c,S = count_divisions(lst, nbits)
        if c > _max:
            _max = c
            #print "found example with", c, "divisions"
            if c == 2**n:
                print "SHUTTER (nbits:",nbits,"n:",n,")"
                print lst
                print S

def go(nbits):
    for n in range(2**nbits):
        find_max_divisions(nbits, n)

def cncpts():
    for i in range(4):
        for j in range(4):
            yield(i, j)

def apply_concept2(lst, C):
    CP, CN = C
    S = set()
    for i in range(len(lst)):
        e = lst[i]
        if e & CP != 0:
            S.add(e)
        if (~e) & CN != 0:
            S.add(e)
    return S

def go2():
    lst = [0,1,3]
    S = {}
    for cncpt in cncpts():
        accptd = apply_concept2(lst, cncpt)
        accptd = tuple(sorted(list(accptd)))
        S[accptd] = cncpt
    return S
