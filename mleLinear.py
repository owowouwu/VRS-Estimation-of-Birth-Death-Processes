# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:58:45 2021

@author: steve
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import Estep as E

eps = 1e-6
# BDP parameters

# birth death sequences
def bRate(k, theta):
    return k * theta[0]

def dRate(k, theta):
    return k * theta[1]

eStep = E.Estep(bRate, dRate)

def loglike(theta, data):
    probs =[]
    # how to avoid for loop here to make the computation faster?
    for i in data:
        a = float(eStep.prob(i, theta))
        if (a <= 0):
            continue
        probs.append(a)
    return -1*np.sum(np.log(probs))

def MLE(guess, data):
    result = minimize(lambda x: loglike(x, data), guess, method = 'L-BFGS-B',
                      bounds = ((1e-6, None),(1e-6, None)))
    print(result)

df1 = pd.read_csv('datalin1.csv', sep=',',header=None, skiprows = 1)
MLE_data = df1.to_records(index=False).tolist()

thetaiter = np.array([0.4, 0.1])

MLE(thetaiter, MLE_data)