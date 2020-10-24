import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class Kernel(object):
    '''
    The Kernel class provides methods for defining kernel functions
    (RBF, periodic, linear) and computing covariances based on those
    functions. 
    '''

    def __init__(self):
        self.RBF, self.per, self.lin = False, False, False
        

    def addKernel(self, ktype, *args):
        '''
        Arguments:
            ktype (string): kernel type ("RBF", "periodic", "linear")
            *args (list of floats): hyperparameters for the specified kernel

        Returns:
            none
        '''
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
        '''
        Arguments:
            x1 (np.array, shape nx1): kernel type ("RBF", "periodic", "linear")
            x2 (np.array, shape mx1): kernel type ("RBF", "periodic", "linear")

        Returns:
            K (np.array, shape nxm): covariance matrix
        '''

        # RBF kernel
        if self.RBF:
            sigma_f = self.sigma_rbf
            L = self.L_rbf

            x_sqdist = cdist(x1, x2, 'sqeuclidean')
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

def buildKernel(params, jitter):
    '''
    Constructs a kernel based on the given parameters. 

    Arguments:
        params (list of floats): kernel parameters
        jitter (float): additional measurement noise for stability purposes

    Returns:
        Kernel object
    '''
   
    # Unpack parameters
    sigma_rbf, L_rbf, sigma_per, L_per, p_per = params

    # RBF kernel - sigma_f, L
    kernel = Kernel()
    kernel.addKernel("RBF", sigma_rbf, L_rbf)
    # kernel.addKernel("RBF", 1, 0.05) # good

    # Periodic kernel - sigma_f, L, p
    kernel.addKernel("periodic", sigma_per, L_per, p_per)

    return kernel


def optimizerFunction(params, X, Y, Xs, jitter):
    '''
    Function to pass to scipy.optimize.minimize in order to compute
    the set of hyperparameters that minimises the log marginal likelihood
    for the given input data

    Arguments:
        params (list of floats): kernel parameters
        X (np.array, size n): input data x values 
        Y (np.array, size n): input data y values
        Xs (np.array, size m): x values to predict for
        jitter (float): additional measurement noise for stability purposes

    Returns:
        val (float): negative of the computed LML
    '''

    kernel = buildKernel(params, jitter)
    mu, sigma, LML = getPosteriorPredictive(X, Y, Xs, kernel, jitter)

    return -LML

def getRMSE(x1, x2):
    '''
    Computes the RMS error between two arrays. 

    Arguments:
        x1 (np.array, size n): data values 
        x2 (np.array, size n): data values 

    Returns:
        val: RMS error between x1 and x2
    '''


    if len(x1) != len(x2):
        print("Could not compute RMSE: unequal input sizes")
        return 0

    return np.sqrt(np.sum(cdist(x1, x2, 'seuclidean')) / len(x1))

def getPosteriorPredictive(X, Y, Xs, kernel, jitter):
    '''
    Computes the posterior predictive distribution and LML
    for a given set of inputs and parameters.

    Arguments:
        X (np.array, size n): input data x values 
        Y (np.array, size n): input data y values
        Xs (np.array, size m): x values to predict for
        kernel (Kernel object): object defining the kernel to be used
        jitter (float): additional measurement noise for stability purposes

    Returns:
        mu (np.array, size m): mean prediction values 
        sigma (np.array, size mxm): predictive covariance
        LML (float): log marginal likelihood of the prediction 
    '''

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

def truncateData(X, Y, time_cutoff):
    '''
    For given data arrays X and Y, truncates both arrays to only include
    data from times before time_cutoff.

    Arguments:
        X (np.array, size n): input data x values 
        Y (np.array, size n): input data y values
        time_cutoff (float): latest time to include in output

    Returns:
        Xc (np.array, size m): truncated x values
        Yc (np.array, size m): truncated y values
    '''

    # Truncate the input datasets according to the time cutoff
    tmax = X[-1]
    for i in range(len(X)):
        if X[i]/tmax >= time_cutoff:
            break

    Xc, Yc = X[:i+1], Y[:i+1]
    return Xc, Yc