#  Steven Nguyen
#  4/12/2020
#  Description - the following program simulates a generalised BDP (birth death process) according a logistic growth model,
#  and extracts discrete observations of population sizes at regular time intervals

import numpy as np
from scipy.optimize import minimize

def dRate(k, theta):
    return k*theta[1]

def logbRate(k, theta):
    return (k**2) * theta[0] * np.exp(-theta[2] * k)

def linbRate(k, theta):
    return k*theta[0]

def dictupdate(dict, pop, update):
    if pop in dict:
        dict[pop] += update
    else:
        dict[pop] = update
    return dict

# runs a birth death process
def bdprocess(maxTime, birthRate, deathRate, theta, start):
    holdingTimes = [0]
    # create n samples of time
    p = [start]
    i = 0
    U = 0
    uk = {}
    D = 0
    dk = {}
    tParticle = 0
    tk = {}
    while (p[i] > 0):
        if (p[i] not in uk.keys()):
            dictupdate(uk, p[i], 0)
        if (p[i] not in dk.keys()):
            dictupdate(dk, p[i], 0)
        if (p[i] not in tk.keys()):
            dictupdate(tk, p[i],0)
        # initialise new birthrates and death rates every transition according to the model in crawford section 2.4.3
        b = birthRate(p[i], theta)
        d = deathRate(p[i], theta)
        # compute the sojourn time
        timeStep = np.random.exponential(1 / (b + d))
        holdingTimes.append(holdingTimes[i] + timeStep)
        if (holdingTimes[i+1] > maxTime):
            tParticle += (maxTime - holdingTimes[i]) * p[i]
            break
        tParticle += timeStep * p[i]
        tk = dictupdate(tk, p[i], timeStep)
        # now we consider whether an individual is born or dies
        unif = np.random.rand()
        if (unif < b / (d + b)):
            uk = dictupdate(uk, p[i], 1)
            p.append(p[i] + 1) # birth
            U+=1
        else:
            dk = dictupdate(dk, p[i], 1)
            p.append(p[i] - 1)
            D+=1
        i += 1
    # return the actual populations and times (useful for plots), and the samples taken every delta timestep
    #print('Death rate: ' +str(D/tParticle))
    #print('Birth rate: ' + str(U/tParticle))
    #print('transitions : ' + str(i))
    #print('Number of up steps %d' %U)
    if (p[i] == 14):
        return np.array([U, D, tParticle])
    return 0

theta = np.array([0.5,0.1])

# compute the gradient of the log likelihood 
def gradQ(theta, uk, tk):
    # partial derivatives
    dLambdaQ = 0
    dBetaQ = 0
    for i in uk.keys():
        dLambdaQ += (uk[i] / theta[0] - tk[i]* i**2 * np.exp(-theta[1]*i))
        dBetaQ += (tk[i]* i**3 * theta[0] * np.exp(-theta[1]*i) - uk[i]*i)
    return np.array([dLambdaQ, dBetaQ])

def hessQ(theta, uk, tk):
    hess = np.zeros((2,2))
    for i in uk.keys(): # change to array of E[Uk|Y] and E[Tk|Y] for implementation in EM algorithm later
        hess[0][0] += -uk[i] / (theta[0]**2)
        hess[0][1] += tk[i] * i**3 * np.exp(-theta[1]*i)
        hess[1][0] = hess[0][1]
        hess[1][1] = -(tk[i] * theta[0] * i**4 * np.exp(-theta[1]*i))
    return hess

def maximise(theta, uk, tk): # this step can stay the same in implementation?
    eps = 1e-6
    thetaprev = theta
    thetanew = theta - np.matmul(np.linalg.inv(hessQ(theta,uk,tk)), gradQ(theta,uk,tk))
    print(thetanew)
    while ((thetanew - thetaprev).all() > eps):
        temp = thetanew
        thetanew = thetanew - np.matmul(np.linalg.inv(hessQ(thetanew,uk,tk)), gradQ(thetanew,uk,tk))
        print(thetanew)
        thetaprev = temp
    return thetanew

def loglike(birth, decay, uk, tk):
    res = 0
    for i in uk.keys():
        res += uk[i]*(np.log(birth) - decay*i) - tk[i]*(birth*i**2*(np.exp(-decay*i)) + i*0.05)
    return res

# define parameters
logtheta = [0.5, 0.3, 0.2]
lintheta = np.array([0.5, 0.3])
thetaiter = np.array([0.1, 0.1])
maxTime = 100
start = np.random.randint(1,21)
nsamples = 50

def simulateE(theta):
    meansum = np.zeros(3)
    n = 0
    for i in range(10000):
        a = bdprocess (1.8553829073952386, linbRate, dRate, theta, 9)
        meansum += a
        n += np.all(a != 0)
    return np.array([meansum[0] / meansum[2], meansum[1] / meansum[2]])
    
while(True):
    print(thetaiter)
    thetaiter = simulateE(thetaiter)

#print(uk)
#print(dk)
#print(tk)

#x = np.outer(np.linspace(0.01, 1, 50), np.ones(50))
#y = x.copy().T # transpose
#f2 = np.vectorize(loglike)
#z = f2(x,y, uk,tk)

#print(z)

#plt.figure()
#ax = plt.axes(projection = '3d')
#ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
#plt.show()
#population, time, samples = bdprocess(maxTime, baseBirth, baseDeath, decay, start, nsamples)

#plt.step(time, population)
#plt.show()
