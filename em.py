# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:13:01 2020

@author: Steven
"""

import numpy as np
from multiprocessing import Pool
from optimparallel import minimize_parallel
from scipy.optimize import minimize
import Estep as E
import pandas as pd

eps = 1e-6
# BDP parameters

# birth death sequences
def dRate(k, theta):
    return k*theta[1]

def bRate(k, theta):
    return (k**2) * theta[0] * np.exp(-theta[2] * k)

def dRate2(k, theta):
    return k*theta[1]

def bRate2(k, theta):
    return k*theta[0]

eStep = E.Estep(bRate, dRate)
eStep2 = E.Estep(bRate2, dRate2)

def loglike(theta, data):
    probs =[]
    #probs = a_pool.map(lambda x: eStep.prob(x, theta), data)
    #probs = [float(eStep.prob(i, theta)) for i in data]
    for i in data:
        a = float(eStep.prob(i, theta))
        if (a <= 0):
            continue
        probs.append(a)
    return -1*np.sum(np.log(probs))

def surrogate(theta, Y, theta0):
    k = Y[0] - 1
    s = 1
    result = 0
    while (k >= 0 and abs(s) > eps):
        #print(str(k))
        s = (eStep.euk(Y, k, theta0) * (np.log(theta[0]) - theta[2] * k) +
                   eStep.edk(Y, k, theta0) * np.log(theta[1]) - 
                   eStep.etk(Y, k, theta0) * (bRate(k, theta) + dRate(k, theta)))
        result += s
        k -= 1
    for k in range(Y[0], Y[1] + 1):
        #print(str(k))
        result += (eStep.euk(Y, k, theta0) * (np.log(theta[0]) - theta[2] * k) +
                   eStep.edk(Y, k, theta0) * np.log(theta[1]) - 
                   eStep.etk(Y, k, theta0) * (bRate(k, theta) + dRate(k, theta)))
    while (abs(s) > eps):
        #print(str(k))
        result += (eStep.euk(Y, k, theta0) * (np.log(theta[0]) - theta[2] * k) +
                   eStep.edk(Y, k, theta0) * np.log(theta[1]) - 
                   eStep.etk(Y, k, theta0) * (bRate(k, theta) + dRate(k, theta)))
        k += 1
    return -1*result

def gradTerm(theta, Y, k):
    gradterm = np.ones(2)
    gradterm[0] = ((eStep.euk(Y, k, theta) / theta[0]) - 
                    (k**2 * np.exp(-theta[2]*k)*eStep.etk(Y,k,theta)))
    gradterm[1] = (theta[0] * k**3 * np.exp(-theta[2]*k) * eStep.etk(Y,k, theta) 
                   - k * eStep.euk(Y,k,theta))
    return gradterm
    
def gradQ(theta, Y):
    k = Y[0] - 1
    g = np.ones(2)
    grad = np.zeros(2)
    while (k >= 0 and np.all((np.abs(g) > eps))):
        #print(g)
        g = gradTerm(theta, Y, k)
        grad += g
        k -= 1
    for k in range(Y[0], Y[1] + 1):
        grad += gradTerm(theta, Y, k)
    k = Y[1] + 1
    g = np.ones(2)
    while(np.all((np.abs(g) > eps))):   
        g = gradTerm(theta, Y, k)
        grad += g
        k += 1
    return grad

def hessTerm(theta, Y, k):
    h = np.ones([2,2])
    h[0][0] = -1*eStep.etk(Y,k,theta) / theta[0]**2
    h[0][1] = k**3 * np.exp(-theta[2]*k) * eStep.etk(Y,k,theta)
    h[1][0] = h[0][1]
    h[1][1] = -theta[0] * k**4 * np.exp(-theta[2]*k) * eStep.etk(Y,k,theta)
    return h

def hessQ(theta, Y):
    k = Y[0] - 1
    h = np.ones([2,2])
    hess = np.zeros([2,2])
    while (k >= 0 and np.all((np.abs(h) > eps))):
        #print('k = %d' % k)
        h = hessTerm(theta, Y, k)
        hess += h
        k -= 1
    for k in range(Y[0], Y[1] + 1):
        hess += hessTerm(theta, Y, k)
        #print('k = %d' % k)
    k = Y[1] + 1
    h = np.ones([2,2])
    while(np.all((np.abs(h) > eps))):
        #print('k = %d' % k)
        h = hessTerm(theta, Y, k)
        #print(h)
        hess += h
        k += 1
    return hess

# m step iteration using equations (22) and (17b)

def mStep(theta, Y):
    thetanew = np.zeros(3)
    hess = hessQ(theta, Y)
    grad = gradQ(theta, Y)
    print('Hessian Matrix:')
    print(hess)
    #check negative definiteness
    if (np.any(np.linalg.eigvals(hess) > 0)):
        print('Warning: non negative definite Hessian found')
    # death rate iteration
    thetanew[1] = eStep.EDdET(Y, theta)
    print('New death rate: %lf' % thetanew[1])
    #print('Gradient:')
    #print(grad)
    # birth rate and exponential parameter iteration
    iterate = np.array([theta[0], theta[2]])
    iterate = iterate - np.matmul(np.linalg.inv(hess), grad)
    thetanew[0], thetanew[2] = iterate[0], iterate[1]
    return thetanew

def EM(theta, Y, maxiter = 50):
    thetanext = mStep(theta, Y)
    it = 0
    while (np.all(np.abs(thetanext - theta) > eps) and it <= maxiter):
        temp = thetanext
        thetanext = mStep(thetanext, Y)
        theta = temp
        it += 1
        print('New theta: ' + str(thetanext))
        if (np.any(thetanext < 0)):
            print('Error, negative value for theta found.')
            return
    if (it > maxiter):
        print('Maximum number of iterations reached.')
        print('Current theta: ' + str(theta))
        return
    print('Estimate for theta:' + str(thetanext))
    print('Number of iterations: ' + str(it))
    return thetanext
    

def theta0(thetareal):
    thetanew = np.ones(3)
    thetanew[0] = thetareal[0] + np.random.uniform(-0.2, 0.2)
    thetanew[1] = thetareal[1] + np.random.uniform(-0.015, 0.015)
    thetanew[2] = thetareal[2] + np.random.uniform(-0.2, 0.2)
    # if any values of theta are negative recalculate
    while (np.any(thetanew < 0)): 
        thetanew[0] = thetareal[0] + np.random.uniform(-0.2, 0.2)
        thetanew[1] = thetareal[1] + np.random.uniform(-0.015, 0.015)
        thetanew[2] = thetareal[2] + np.random.uniform(-0.2, 0.2)

    return np.array(thetanew)

#thetareal = [0.5, 0.05, 0.8]

datafile = 'data2.csv'
df0 = pd.read_csv(datafile, sep=',',header=None, skiprows = 1)
data = df0.to_records(index=False).tolist()
datapoint = data[3]
x0 = np.array([0.34352752, 0.05719226, 0.30942919])
theta_iter = x0


for _ in range(10):
    a= minimize(lambda x: surrogate(x, datapoint, theta_iter), x0, method = 'L-BFGS-B',
             bounds = ((1e-6, None),(1e-6, None),(1e-6, None)))
    print(a)
    theta_iter = a.x

#EM(x0, datapoint)