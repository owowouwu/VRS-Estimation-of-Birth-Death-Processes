# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:13:01 2020

@author: Steven
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

def mStep(theta, Y):
    return np.array([(eStep.EUdET(Y, theta)), (eStep.EDdET(Y, theta))])

def mStep2(theta, data):
    u = 0
    t = 0
    d = 0
    for i in data:
        u += eStep.EU(i, theta)
        d += eStep.ED(i, theta)
        t += eStep.ET(i, theta)
    return np.array([u/t, d/t])

def surrogate(theta, Y, theta0):
    k = Y[0] - 1
    s = 1
    result = 0
    while (k >= 0 and abs(s) > eps):
        #print(str(k))
        s = (eStep.euk(Y, k, theta0) * np.log(k*theta[0]) +
             eStep.edk(Y, k, theta0) * np.log(k*theta[1]) -
             eStep.etk(Y, k, theta0) * k * (theta[0] + theta[1]))
        result += s
        k -= 1
    for k in range(Y[0], Y[1] + 1):
        #print(str(k))
        result += (eStep.euk(Y, k, theta0) * np.log(k*theta[0]) +
                   eStep.edk(Y, k, theta0) * np.log(k*theta[1]) -
                   eStep.etk(Y, k, theta0) * k * (theta[0] + theta[1]))
    while (abs(s) > eps):
        #print(str(k))
        result += (eStep.euk(Y, k, theta0) * np.log(k*theta[0]) +
                   eStep.edk(Y, k, theta0) * np.log(k*theta[1]) -
                   eStep.etk(Y, k, theta0) * k * (theta[0] + theta[1]))
        k += 1
    return -1*result

def EM(theta, Y, maxiter = 200):
    thetanext = mStep(theta, Y)
    it = 0
    while (np.all(np.abs(thetanext - theta) > eps) and it <= maxiter):
        temp = thetanext
        thetanext = mStep(thetanext, Y)
        theta = temp
        it += 1
        print('Theta = ' + str(thetanext))
        if (np.any(thetanext < 0)):
            print('Error, negative value for theta found.')
            return
    if (it == maxiter): 
        print('Error: maximum number of iterations reached.')
        return
    print('Estimate for theta:' + str(thetanext))
    print('Number of iterations: ' + str(it))
    return thetanext

def EM2(theta, data, maxiter = 200):
    thetanext = mStep2(theta, data)
    it = 0
    while (np.all(np.abs(thetanext - theta) > eps) and it <= maxiter):
        temp = thetanext
        thetanext = mStep2(thetanext, data)
        theta = temp
        it += 1
        print('Theta = ' + str(thetanext))
        if (np.any(thetanext < 0)):
            print('Error, negative value for theta found.')
            return
    if (it == maxiter): 
        print('Error: maximum number of iterations reached.')
        return
    print('Estimate for theta:' + str(thetanext))
    print('Number of iterations: ' + str(it))
    return thetanext

# MLE directly using probabilities

def loglike(theta, data):
    probs =[]
    # how to avoid for loop here to make the computation faster?
    for i in data:
        a = float(eStep.prob(i, theta))
        if (a <= 0):
            continue
        probs.append(a)
    return -1*np.sum(np.log(probs))

def sanityCheck(Y, theta):
    print('Expected births: %lf' % eStep.EU(Y, theta))
    print('Expected deaths: %lf' % eStep.ED(Y, theta))
    print('Expected particle time: %lf ' % eStep.ET(Y, theta))
    print('ED / ET method 1: %lf' % eStep.EDdET(Y, theta))
    print('EU / ET method 1: %lf' % eStep.EUdET(Y, theta))
    
df0 = pd.read_csv('datalin.csv', sep=',',header=None, skiprows = 1)
EM_data = df0.to_records(index=False).tolist()
df1 = pd.read_csv('datalinMLE.csv', sep=',',header=None, skiprows = 1)
MLE_data = df1.to_records(index=False).tolist()

thetareal = np.array([0.5, 0.3])
thetaiter = np.array([0.27274981, 0.026302  ])

# testing functions
theta = np.array([0.5, 0.3])
Y = [19, 27, 1]
#print((mStep2(thetareal, MLE_data[0:10])))
EM2(thetareal, MLE_data[0:10])
theta_ests = []
#sanityCheck(datapoint, thetareal)
#for i in EM_data:
#    print('Using a = %d, b = %d, t = %lf' % (i[0], i[1], i[2]))
#    a = EM(thetareal, i)
#    theta_ests.append(a)

#a = minimize(lambda x: surrogate(x, EM_data[0], thetaiter), thetaiter,
#             method = 'L-BFGS-B', bounds = ((1e-6, None), (1e-6, None)))
#print(a)

