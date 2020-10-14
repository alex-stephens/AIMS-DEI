import numpy as np
import pylab

# Based on https://gist.github.com/stober/4964727

class Kernel(object):

    def __init__(self, ktype="RBF",*args):
        self.type = ktype
        self.thetas = args

    def __call__(self,x1, x2):

        if self.type == "RBF":
            sigma_f = self.thetas[0]
            L = self.thetas[1]

            K = np.zeros((len(x1), len(x2)))
            for i,x in enumerate(x1):
                for j,y in enumerate(x2):
                    K[i,j] = sigma_f**2 * np.exp( - 0.5 * ( ((x - y)/L)**2) )
            return K


            # return sigma_f**2 * np.exp( - 0.5 * ( ((x1 - x2)/L)**2) )

        if self.type == "RBF2":
            sigma_f = self.thetas[0]
            L = self.thetas[1]

            return sigma_f**2 * np.exp( - 0.5 * ( ((x1 - x2)/L)**2) )

        elif self.type == "periodic":
            sigma_f = self.thetas[0]
            L = self.thetas[1]

            # TODO
            return sigma_f**2 * np.exp( - 0.5 * ( ((x1 - x2)/L)**2) )

        elif self.type == "linear":
            sigma_f = self.thetas[0]
            c = self.thetas[1]
            
            # TODO
            return sigma_f**2 * np.exp( - 0.5 * ( ((x1 - x2)/L)**2) )

def constructCovariance(x,kernel):
    K = np.reshape([kernel(x1,x2) for x1 in x for x2 in x], (len(x),len(x)))
    return K

def GetPosteriorPredictive(X, Y, Xs, K, sigma_n):

    k = kernel(Xs, X)
    Kinv = np.linalg.inv(K)

    # TODO: add sigma_n term

    mu = np.matmul(np.matmul(k,Kinv),Y)
    sigma = kernel(Xs, Xs) - np.matmul(np.matmul(k,Kinv),np.transpose(k))

    return (mu, sigma)

def GetFunctionSample(mu, sigma):
    fs = np.random.multivariate_normal(mu, sigma, 1)
    return np.transpose(fs)

# RBF kernel - sigma_f, L
kernel = Kernel("RBF", 25.0, 0.5)

points = 5
X = np.random.rand(points)

Y = np.random.rand(points) * 2 - 1
sigma_n = 0

# mmean and covariance functions
mu = np.zeros(len(X))
K = kernel(X, X)
print(K)

Xs = np.linspace(-5, 5, 100)
mu, sigma = GetPosteriorPredictive(X, Y, Xs, K, sigma_n)

y1 = np.array([mu[i] + 2 * sigma[i][i] for i in range(len(Xs))])
y2 = np.array([mu[i] - 2 * sigma[i][i] for i in range(len(Xs))])
pylab.fill_between(Xs, y1, y2, linewidth=0, color='blue', alpha=0.1)
y1 = np.array([mu[i] + 1 * sigma[i][i] for i in range(len(Xs))])
y2 = np.array([mu[i] - 1 * sigma[i][i] for i in range(len(Xs))])
pylab.fill_between(Xs, y1, y2, linewidth=0, color='blue', alpha=0.2)


draws = 10
for i in range(draws):
    fs = GetFunctionSample(mu, sigma)
    pylab.plot(Xs, fs, color='black', alpha=1.0)

pylab.plot(X, Y, 'rx')

# pylab.errorbar(Xs, mu, yerr=sigma)

pylab.show()
