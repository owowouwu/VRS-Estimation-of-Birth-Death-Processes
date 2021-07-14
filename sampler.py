#  Steven Nguyen
#  4/12/2020
#  Description - the following program simulates a BDP (birth death process) according a logistic growth model,
#  and extracts discrete observations of population sizes at regular time intervals

import numpy as np
import pandas as pd

# define parameters

def dRate(k, theta):
    return k*theta[1]

def logbRate(k, theta):
    return (k**2) * theta[0] * np.exp(-theta[2] * k)

def linbRate(k, theta):
    return k*theta[0]

# runs a birth death process for sampling described by crawford (section 3.2)
def bdprocess(maxTime, bRate, dRate, theta, start):
    holdingTimes = [0]
    # create n samples of time
    p = [start]
    i = 0
    while (p[i] > 0):
        # initialise new birthrates and death rates every transition according to the model in crawford section 2.4.3
        deathRate = dRate(p[i], theta)
        birthRate= bRate(p[i], theta)
        # similar to the above process
        timeStep = np.random.exponential(1 / (birthRate + deathRate))
        holdingTimes.append(holdingTimes[i] + timeStep)
        if (holdingTimes[i + 1] > maxTime):
            break
        # now we consider whether an individual is born or dies
        unif = np.random.rand()
        if (unif < birthRate / (deathRate + birthRate)):
            p.append(p[i] + 1) # birth
        else:
            p.append(p[i] - 1)
        i += 1
    sample = (start, p[i], maxTime)
    return sample

def bdsample(N, bRate, dRate, theta, output):
    samples = []
    for i in range(N):
        start = np.random.randint(1,21)
        sampletime = np.random.uniform(0.1,3)
        samples.append(bdprocess(sampletime, bRate, dRate, theta, start))
    df = pd.DataFrame(samples)
    df.to_csv(output, index = False)
    return 

# sampling by observing the BDP at discrete times

def bdprocess2(maxTime, bRate, dRate, theta, start, N):
    holdingTimes = [0]
    p = [start]
    i = 0
    #print('Starting point: ' +str(p[i]) +str(', t= ') + str(holdingTimes[i]))
    while (p[i] > 0) and (holdingTimes[i] < maxTime):
        # initialise new birthrates and death rates every transition according to the model in crawford section 2.4.3
        deathRate = dRate(p[i], theta)
        birthRate= bRate(p[i], theta)
        # similar to the above process
        timeStep = np.random.exponential(1 / (birthRate + deathRate))
        holdingTimes.append(holdingTimes[i] + timeStep)
        # now we consider whether an individual is born or dies
        unif = np.random.rand()
        if (unif < birthRate / (deathRate + birthRate)):
            p.append(p[i] + 1) # birth
        else:
            p.append(p[i] - 1)
        i += 1
    i = 0
    elapsed = 0
    #print(len(p))
    # go through and observe the BDP N times
    timeDelta = maxTime / (N - 1)
    sampleP = [start]
    s = 0
    data = []
    while (i < len(p) and elapsed < maxTime):
        #print('Elapsed time: %lf' % elapsed )
        #print(str(i))
        elapsed += timeDelta
        #print('Current time and pop: ' + str(p[i]) +str(', t= ') + str(holdingTimes[i]))
        while (elapsed > holdingTimes[i] and i < len(p) - 1):
            #print(i)
            i += 1
            #print('Going through : '+ str(p[i]) +str(', t= ') + str(holdingTimes[i]))
        sampleP.append(p[i - 1])
        s += 1
        a = (sampleP[s-1], sampleP[s], timeDelta)
        data.append(a)
    data.append((sampleP[s], p[-1], timeDelta))
    return data

def flattenList(arr):
    return [item for sublist in arr for item in sublist]

# simulate M trajectories of a BDP, sampling each one discretely n times at different time intervals
def bdsample2(n, M, maxTime, bRate, dRate, theta, output):
    data = []
    for i in range(M):
        # random starting point anywhere
        start = np.random.randint(1,21)
        data.append(bdprocess2(maxTime, bRate, dRate, theta, start, n))
    data = flattenList(data)
    df = pd.DataFrame(data)
    df.to_csv(output, index = False)
    return

theta = [0.3, 0.05, 0.5]
theta_lin = [0.5, 0.2]
maxTime = 100

#bdsample(50, 100, logbRate, dRate, theta, './data3.csv')
#bdsample2(20, 10, 100, logbRate, dRate, theta, './dataMLE3.csv')

#bdsample2(50, 1, 10, linbRate, dRate, theta_lin, 'datalinMLE.csv')

for i in range(100,200):
    bdsample2(200, 1, 1000, logbRate, dRate, theta, './DataMLE3/data%d.csv' % i)    
