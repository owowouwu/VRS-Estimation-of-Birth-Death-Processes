# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 23:15:12 2021

@author: Steven
"""

import numpy as np
from optimparallel import minimize_parallel
import Estep as E
import pandas as pd
import time

def dRate(k, theta):
    return k*theta[1]

def bRate(k, theta):
    return (k**2) * theta[0] * np.exp(-theta[2] * k)

thetareal = [0.3, 0.05, 0.5]

eStep = E.Estep(bRate, dRate)

def loglike(theta, data):
    probs =[]
    #probs = a_pool.map(lambda x: eStep.prob(x, theta), data)
    for i in data:
        a = float(eStep.prob(i, theta))
        if (a <= 0):
            continue
        probs.append(a)
    return -1*np.sum(np.log(probs))

datafile = './DataMLE3/data103.csv'
df0 = pd.read_csv(datafile, sep=',',header=None, skiprows = 1)
data = df0.to_records(index=False).tolist()
def f(theta):
    return loglike(theta, data)

def main():
    theta_init = [0.34352752, 0.05719226, 0.30942919]
    #print('Actual loglikelihood at real value: '+str(loglike(thetareal, data)))
    o1 = minimize_parallel(f, theta_init, bounds = ((1e-6, None),(1e-6, None),(1e-6, None)))
    print(o1)
    
if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)