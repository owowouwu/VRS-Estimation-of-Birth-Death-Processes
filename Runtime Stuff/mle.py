# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 23:15:12 2021

@author: Steven
"""

import numpy as np
from optimparallel import minimize_parallel
import Estep as E
import pandas as pd
from scipy.linalg import expm
import time

def dRate(k, theta):
    return k*theta[1]

def bRate(k, theta):
    return (k**2) * theta[0] * np.exp(-theta[2] * k)

thetareal = [0.3, 0.05, 0.5]

eStep = E.Estep(bRate, dRate)

def loglike(theta, data):
    maxState = 500
    timestep = data[0][2]
    tRates = [-(bRate(i, theta) + dRate(i, theta)) for i in range(maxState)]
    dRates = [dRate(i, theta) for i in range(1,maxState)]
    bRates = [bRate(i - 1,theta) for i in range(1,maxState)]    
    rateMatrix = np.diag(tRates) + np.diag(bRates, 1) + np.diag(dRates, -1)
    probs = []
    #probs = a_pool.map(lambda x: eStep.prob(x, theta), data)
    for i in data:
        matExp = expm(rateMatrix * timestep)
        a = matExp[i[0]][i[1]]
        if (a <= 0):
            continue
        probs.append(a)
    return -1*np.sum(np.log(probs))

datafile = 'data103.csv'
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