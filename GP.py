import numpy as np

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

            Krbf = np.zeros((len(x1), len(x2)))

            for i,x in enumerate(x1):
                for j,y in enumerate(x2):
                    Krbf[i,j] = sigma_f**2 * np.exp( - 0.5 * ( ((x - y)/L)**2) )

        # Periodic kernel
        if self.per:
            sigma_f = self.sigma_per
            L = self.L_per
            p = self.p_per

            Kper = np.zeros((len(x1), len(x2)))
            for i,x in enumerate(x1):
                for j,y in enumerate(x2):
                    Kper[i,j] = sigma_f**2 
                    Kper[i,j] *= np.exp( - (2/L**2) * np.sin(np.pi * (x-y)/p)**2 )

        # Linear kernel
        if self.lin:
            sigma_f = self.sigma_lin
            c = self.c_lin

            Klin = np.zeros((len(x1), len(x2)))
            for i,x in enumerate(x1):
                for j,y in enumerate(x2):
                    Klin[i,j] = sigma_f**2 * (x-c) * (y-c)


        # return Kper
        # return Krbf
        # return np.multiply(Krbf,Kper)
        return Krbf + Kper

def GetPosteriorPredictive(X, Y, Xs, kernel, jitter):

    K = kernel(X, X)
    Ks = kernel(X, Xs)
    Kss = kernel(Xs, Xs)

    L = np.linalg.cholesky(K + jitter**2 * np.eye(len(X)))
    Linv = np.linalg.inv(L)

    alpha = np.dot(Linv.T, np.dot(Linv, np.transpose(Y)))
    mu = np.dot(Ks.T, alpha)
    
    v = np.dot(Linv, Ks)
    sigma = Kss - np.dot(v.T, v)

    return (mu, sigma)

def GetFunctionSample(mu, sigma):
    fs = np.random.multivariate_normal(mu, sigma, 1)
    return np.transpose(fs)
