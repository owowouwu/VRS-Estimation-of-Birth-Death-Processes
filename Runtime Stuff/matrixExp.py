import numpy as np
from scipy.linalg import expm

# used for checking transition probabilities

def dRate(k, base):
    return k*base

def bRate(k, base, alpha):
    return (k**2) * base * np.exp(-alpha * k)

birth = 0.3
decay = 0.5
death = 0.05

theta = [birth, death, decay]

def matrixprob(Y, theta, maxState=100):
    tRates = [-(bRate(i, theta[0], theta[2]) + dRate(i, theta[1])) for i in range(maxState)]
    dRates = [dRate(i, theta[1]) for i in range(1,maxState)]
    bRates = [bRate(i - 1, theta[0], theta[2]) for i in range(1,maxState)]    
    rateMatrix = np.diag(tRates) + np.diag(bRates, 1) + np.diag(dRates, -1)
    matExp = expm(rateMatrix * Y[2])
    return matExp[Y[0]][Y[1]]

Y = (10, 12, 1)

matrixprob(Y, theta)

