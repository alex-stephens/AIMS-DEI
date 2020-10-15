import numpy as np
from scipy.spatial.distance import cdist

class Kernel(object):

    def __init__(self):
        self.RBF, self.per, self.lin = False, False, False
        

    def addKernel(self, ktype, *args):
        
        if ktype == "RBF":
            self.RBF = True
            self.sigma_rbf = args[0]
            self.L_rbf = args[1]

        elif ktype == "periodic":
            self.per = True
            self.sigma_per = args[0]
            self.L_per = args[1]
            self.p_per = args[2]

        elif ktype == "linear":
            self.lin = True
            self.sigma_lin = args[0]
            self.c_lin = args[1]

        else:
            print("Invalid kernel type")


    def __call__(self, x1, x2):

        # RBF kernel
        if self.RBF:
            sigma_f = self.sigma_rbf
            L = self.L_rbf

            x_sqdist = cdist(x1, x2, 'seuclidean')
            Krbf = sigma_f**2 * np.exp(- 0.5 * (x_sqdist) / (L**2))

        # Periodic kernel
        if self.per:
            sigma_f = self.sigma_per
            L = self.L_per
            p = self.p_per

            x_dist = cdist(x1, x2, 'euclidean')
            Kper = sigma_f**2 * np.exp(-2/(L**2) * np.sin(np.pi * x_dist/p)**2)

        # Linear kernel
        if self.lin:
            sigma_f = self.sigma_lin
            c = self.c_lin
            
            x_dist = cdist(x1, x2, 'euclidean')
            Klin = sigma_f**2 * (x1-c).T * (x2-c)
        
        return Krbf + Kper

def buildKernel(params):
    
    # Unpack parameters
    jitter, sigma_rbf, L_rbf, sigma_per, L_per, p_per = params

    # RBF kernel - sigma_f, L
    kernel = Kernel()
    kernel.addKernel("RBF", sigma_rbf, L_rbf)
    # kernel.addKernel("RBF", 1, 0.05) # good

    # Periodic kernel - sigma_f, L, p
    kernel.addKernel("periodic", sigma_per, L_per, p_per)

    return kernel


def optimizerFunction(params, X, Y, Xs):

    kernel = buildKernel(params)
    jitter = params[0]

    mu, sigma, LML = getPosteriorPredictive(X, Y, Xs, kernel, jitter)

    print("LML:", LML)

    return -LML

def getPosteriorPredictive(X, Y, Xs, kernel, jitter):

   # Covariance matrices
    K = kernel(X, X)
    Ks = kernel(X, Xs)
    Kss = kernel(Xs, Xs)

    # Cholesky decomposition
    L = np.linalg.cholesky(K + jitter**2 * np.eye(len(X)))
   
    Linv = np.linalg.inv(L)
    alpha = np.dot(Linv.T, np.dot(Linv, Y))
    v = np.dot(Linv, Ks)
    
    # Predictive mean and variance
    mu = np.dot(Ks.T, alpha)
    sigma = Kss - np.dot(v.T, v)

    # Log marginal likelihood
    LML = float(-0.5 * np.dot(Y.T, alpha) - sum([np.log(L[i,i]) for i in range(len(L))]) - 0.5 * len(X) * np.log(2 * np.pi))

    return (mu, sigma, LML)