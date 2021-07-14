import numpy as np
import mpmath as mp
import time
tiny = 1e-30
eps = 1e-6

def invLaplace(f, t):
    # arbitrary truncation?
    J = 49
    A = 20
    s0 = [np.exp(A / 2) * f(A/(2*t)) / (2*t)]
    # sequence to sum using levin
    sN = [np.exp(A/2)/t * ((-1)**k) * f((A + (2*k*np.pi)*1j) / (2*t)).real  
        for k in range(1, J+1)]
    s = s0 + sN
    #print(str(sN))
    # if all 0s, then return 0
    if (np.all(np.abs(s) == 0)):
        return 0
    L = mp.levin(method = "levin", variant = "u")
    return float(L.update(s)[0])

# class for EM algorithm
class Estep:
    def __init__(self, bRate, dRate):
        self.bRate = bRate
        self.dRate = dRate
    
    # continued fraction terms
    def an(self, k, theta):
        if (k == 1):
            return 1
        return -1*self.bRate(k - 2, theta) * self.dRate(k - 1, theta)
     
    def bn(self, k, s, theta):
        if (k == 0):
            return 0
        if (k == 1):
            return s 
        return s + self.bRate(k-1, theta) + self.dRate(k - 1, theta)

    # compute the ratio Bk-1 / Bk using Lentz method
    def Dk(self, k,s, theta):
        depth = 1
        if (k == 1):
            return(1 / self.bn(1, s, theta))
        Dk = self.bn(depth - 1,s, theta) / self.bn(depth,s, theta)
        while (depth < k):
            depth += 1
            Dk = self.bn(depth,s, theta) + self.an(depth, theta) * Dk
            if (Dk == 0):
                Dk = tiny
            Dk = 1 / Dk
        return Dk

    # compute the ratio Bi / Bj for i > j
    def BidBj(self, s,i,j, theta):
        a = min(i,j)
        b = max(i,j)
        if (i == j):
            return 1
        if (b == (a + 1)):
            return self.Dk(b,s, theta)
        #print('Current values (%lf + i%lf, %d, %d), a1 = %lf' % (s.real, s.imag, i ,j,self.Dk(a + 1, s, theta)))
        a2 = 1
        a1 = 1 / self.Dk(a + 1, s, theta)
        k = a + 2
        while (k <= b):
            res = self.bn(k,s, theta) * a1 + self.an(k, theta) * a2
            a2 = a1
            a1 = res
            k += 1
        return 1/res

    # compute the continued fraction component using Lentz method and error bound described in Crawford
    def contFrac(self, k,s, theta):
        eps = 1e-8
        fnext = tiny
        f = fnext
        Ck = tiny
        Dk = 0
        error_bound = 1
        depth = 1
        while (error_bound > eps):
            Dk = self.bn(k + depth, s, theta) + self.an(k + depth, theta) * Dk
            Ck = self.bn(k + depth, s, theta) + self.an(k + depth, theta) / Ck
            if (Dk == 0):
                Dk = tiny
            if (Ck == 0):
                Ck = tiny
            Dk = 1 / Dk
            fnext = f * Ck * Dk
            diff = abs(fnext - f)
            if (diff == 0):
                diff = tiny
            if (Dk.imag == 0):
                error_bound = abs(Ck*Dk - 1)
            else:
                error_bound = abs(1/Dk) * diff / abs((1/Dk).imag)
            f = fnext
            depth += 1
        #print('Frac = ' + str(f))
        return f

    # compute f_ij (s)
    def lapProb(self, i, j, s, theta):
        if (j <= i):
            if (j == i):
                prod = 1
            else:
                prod = np.prod([self.dRate(x, theta) for x in range(j + 1 , i + 1)])
            B1 = self.BidBj(s, i,j, theta)
            B2 = 1 / self.Dk(i + 1, s, theta)
            result = prod * B1 / (B2 + self.contFrac(i + 1, s, theta))
        if (i < j):
            prod = np.prod([self.bRate(x, theta) for x in range(i, j)])
            B1 =  self.BidBj(s, i,j, theta)
            B2 = 1 / self.Dk(j+1, s, theta)
            result= prod * B1 / (B2 + self.contFrac(j + 1, s, theta))
        #print('B1 = ' + str(B1))
        #print('B2 = ' + str(B2))
        #print('product term = %lf' % prod)
        #print('f_%d%d (%lf + %lfi) = %lf + %lfi' % (i,j,s.real, s.imag, result.real, result.imag))
        return result
    
    # Y here is a vector containing [initial state, final state, time]
    def prob(self, Y, theta):
        return invLaplace(lambda x: self.lapProb(Y[0],Y[1], x, theta), Y[2])
    
    # expectations of ups/downs/time at state k, time t
    def euk(self, Y, k, theta):
        conv = self.bRate(k, theta)*invLaplace(lambda x: self.lapProb(Y[0],k,x, theta)
                                               *self.lapProb(k+1,Y[1],x, theta),Y[2])
        return conv / self.prob(Y, theta)
    
    def edk(self, Y, k, theta):
        conv = self.dRate(k, theta)*invLaplace(lambda x: self.lapProb(Y[0],k,x, theta)
                                               *self.lapProb(k-1,Y[1],x, theta), Y[2])
        return conv / self.prob(Y, theta)
    
    def etk(self, Y, k, theta):
        #print('f_%d,%d (1) = %lf' %(Y[0], k, self.lapProb(Y[0],k,1, theta)))
        #print('f_%d,%d (1) = %lf' %(k, Y[1], self.lapProb(k,Y[1],1, theta)))
        conv = invLaplace(lambda x: self.lapProb(Y[0],k,x, theta)
                          *self.lapProb(k,Y[1],x,theta),Y[2])
        return conv / self.prob(Y, theta)
    
    # numerical summation of laplace transforms 'f' given a and b
    def truncSum(self,f, a,b, s):
        low = min(a,b)
        high = max(a,b)
        k = low - 1
        res = 0
        term = 1
        # now sum the terms from i to 0, stopping if the terms become insignificant
        while (k >= 0 and abs(term) > eps):
            term = f(k,s)
            #print('term : ' +str(term))
            res += term
            k -= 1
        for k in range(low, high + 1):
            res += f(k,s)
        k = high + 1
        term = 1
        while (abs(term) > eps):
            term = f(k,s)
            #print('term : ' +str(term))
            res += term
            k +=  1
        return res
    
    # laplace transforms of expectation of total up/down/particle time
    def EU(self, Y, theta):
        # laplace transform of euk
        def fuk(k, s):
            return self.bRate(k, theta) * self.lapProb(Y[0],k ,s, theta) * self.lapProb(k+1,Y[1],s, theta)
        return invLaplace(lambda x: self.truncSum(fuk,Y[0],Y[1],x), Y[2]) / self.prob(Y, theta)
   
    def ED(self, Y, theta):
        # laplace transform of edk
        def fdk(k,s):
            return self.dRate(k, theta) * self.lapProb(Y[0],k,s, theta) * self.lapProb(k-1,Y[1],s, theta)
        return invLaplace(lambda x: self.truncSum(fdk, Y[0],Y[1],x), Y[2]) / self.prob(Y, theta)
   
    def ET(self, Y, theta):
        # laplace transform of etk
        def ftk(k,s):
            return k*self.lapProb(Y[0],k,s, theta)*self.lapProb(k,Y[1],s, theta)
        return invLaplace(lambda x: self.truncSum(ftk, Y[0],Y[1],x), Y[2]) / self.prob(Y, theta)
    
    # useful for calculating iterates for lienar model, as no need to calculate transition probabilities
    
    def EDdET(self, Y, theta):
        def fdk(k,s):
            return self.dRate(k, theta) * self.lapProb(Y[0],k,s, theta) * self.lapProb(k-1,Y[1],s, theta)
        def ftk(k,s):
            return k*self.lapProb(Y[0],k,s, theta)*self.lapProb(k,Y[1],s, theta)
        conv1 = invLaplace(lambda x: self.truncSum(fdk,Y[0],Y[1],x), Y[2])
        conv2 = invLaplace(lambda x: self.truncSum(ftk, Y[0],Y[1],x),Y[2])
        return conv1 / conv2
    
    def EUdET(self, Y, theta):
        def fuk(k, s):
            return self.bRate(k, theta) * self.lapProb(Y[0],k ,s, theta) * self.lapProb(k+1,Y[1],s, theta)
        def ftk(k,s):
            return k*self.lapProb(Y[0],k,s, theta)*self.lapProb(k,Y[1],s, theta)
        conv1 = invLaplace(lambda x: self.truncSum(fuk,Y[0],Y[1],x), Y[2])
        conv2 = invLaplace(lambda x: self.truncSum(ftk, Y[0],Y[1],x,),Y[2])
        return conv1 / conv2

def dRate(k, theta):
    return k*theta[1]

def bRate(k, theta):
    return (k**2) * theta[0] * np.exp(-theta[2] * k)

birth = 0.3
decay = 0.5
death = 0.05

theta = [birth, death, decay]

logistic = Estep(bRate, dRate)

start = time.time()
logistic.prob((1, 2, 1.32), theta)
end = time.time()
print(end - start)