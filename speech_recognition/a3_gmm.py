import os
import fnmatch
import random
from math import log
from math import exp
from math import pi as PI
from math import e as E
from math import inf as INF
import numpy as np
from scipy.misc import logsumexp
from sklearn.model_selection import train_test_split

dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.ones((M, d))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    d = 13
    mu = myTheta.mu[m]
    sigma = myTheta.Sigma[m]
    result = 0
    for index in range(d):
        result += (1/2 * ((x[index])**2) - mu[index] * x[index]) * (1/sigma[index])
    if preComputedForM != []:
        result = result * (-1) - preComputedForM[m]
    else:
        result = result * (-1) - preCompM(myTheta)[m]

    return result


def log_p_m_x(m, x, myTheta, log_Bs=[]):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    p_nume = 0
    p_deno = 0
    if log_Bs != []:
        p_nume = log(myTheta.omega[m][0], E) + log_Bs[m]
    else:
        preComputedForM = preCompM(myTheta)
        p_nume = log(myTheta.omega[m][0], E) + log_b_m_x(m, x, myTheta, preComputedForM)
        log_Bs = np.zeros(m)
        for index in range(m):
            log_Bs[index] = log_b_m_x(index, x, myTheta, preComputedForM)
    p_deno = logsumexp(log_Bs, b=myTheta.omega.transpose()[0])

    return p_nume - p_deno


def preCompM(myTheta):
    d = 13
    result = []

    for index in range(myTheta.mu.shape[0]):
        mu = myTheta.mu[index]
        sigma = myTheta.Sigma[index]
        compute = 0
        for j in range(d):
            compute += (((mu[j])**2) / (2 * sigma[j])) + log(sigma[j], E)/2
        compute += d/2 * log(2*PI, E)
        result.append(compute)

    return result


def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    result = 0
    log_Bs = log_Bs.transpose()
    for t in range(log_Bs.shape[1]):
        omega = myTheta.omega.transpose()[0]
        result += logsumexp(log_Bs[t], b=omega)
    return result


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    # X is a T * d matrix
    T = X.shape[0]
    # init myTheta
    myTheta = theta(speaker, M, X.shape[1])
    for m in range(M):
        myTheta.omega[m][0] = 1/M
    myTheta.mu = X[np.random.choice(T, M, replace=False)]

    curr_iter = 0
    prev_l = -INF
    improvement = INF
    while curr_iter <= maxIter and improvement > epsilon:
        b_m = np.zeros((M, T)) # M * T matrix
        p_m = np.zeros((M, T))
        for m in range(M):
            for t in range(T):
                b_m[m][t] = log_b_m_x(m, X[t], myTheta)
        log_bs = b_m.transpose()
        for m in range(M):
            for t in range(T):
                p_m[m][t] = log_p_m_x(m, X[t], myTheta, log_bs[t])
        # compute L
        L = exp(logLik(b_m, myTheta))
        # update parameter
        d = 13
        new_omega = np.zeros((M, 1))
        new_mu = np.zeros((M, d))
        new_sigma = np.ones((M, d))
        for m in range(M):
            omega = 0
            mu = np.zeros(d)
            sigma = np.zeros(d)
            for t in range(T):
                omega += exp(p_m[m][t])
                mu = np.add(mu, np.multiply(X[t], exp(p_m[m][t])))
                sigma = np.add(sigma, np.multiply(np.square(X[t]), exp(p_m[m][t])))
            new_omega[m][0] = omega / T
            mu = np.divide(mu, omega)
            new_mu[m] = mu
            sigma = np.divide(sigma, omega)
            sigma = np.subtract(sigma, np.square(mu))
        myTheta.omega = new_omega
        myTheta.mu = new_mu
        myTheta.Sigma = new_sigma
        improvement = L - prev_l
        prev_l = L
        curr_iter += 1
    print('curr_iter: ' + str(curr_iter))
    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    method = ''
    print(models[correctID].name)
    logs = []
    T = len(mfcc)
    M = len(models[0].omega)
    for model in models:
        b_m = np.zeros((M, T)) # M * T matrix
        for m in range(M):
            for t in range(T):
                b_m[m][t] = log_b_m_x(m, mfcc[t], model)
        logs.append(logLik(b_m, model))
    bestModel = logs.index(max(logs))
    for _ in range(k):
        idx = logs.index(max(logs))
        log_like = logs.pop(idx)
        print(models[idx].name, str(log_like))
    print('\n')
    return 1 if (bestModel == correctID) else 0

def train_test(M, epsilon):
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    maxIter = 20
    trainThetas = []
    testMFCCs = []
    print('M: ' + str(M) + ', epsilon: ' + str(epsilon))
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)
            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)
            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)
            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)
            trainThetas.append(train(speaker, X, M, epsilon, maxIter))
    # evaluate
    for speaker_count in [32, 16, 10]:
        print("number of speakers: " + str(speaker_count))
        numCorrect = 0
        for i in range(0, len(testMFCCs)):
            numCorrect += test(testMFCCs[i], i, trainThetas, k)
        accuracy = 1.0*numCorrect/len(testMFCCs)
        print('Accuracy: ' + str(accuracy) + '\n\n')

if __name__ == "__main__":
    M_list = [8, 4]
    epsilon_list = [0.0, 0.1]
    for M in M_list:
        for epsilon in epsilon_list:
            train_test(M, epsilon)
