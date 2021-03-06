import pylab
import numpy as np

def getFunctionSample(mu, sigma):
    '''
    Sample from a multivariate Gaussian distribution.

    Arguments:
        mu (np.array, size n): mean vector
        sigma (np.array, size nxn): covariance matrix

    Returns:
        fs (np.array, size n): sampled vector
    '''
    fs = np.random.multivariate_normal(mu, sigma, 1)
    return np.transpose(fs)


def plotTides(Xs, mu, sigma, t, y, ttrue, ytrue, time_cutoff=None, title=None):
    '''
    Visualise the given tide height data and GP predictions..

    Arguments:
        Xs (np.array, size m): x values to predict for
        mu (np.array, size n): mean vector
        sigma (np.array, size nxn): covariance matrix
        t (np.array): time values for training data
        y (np.array): tide height values for training data
        ttrue (np.array): time values for test data
        ytrue (np.array): tide height values for test data
        time_cutoff (float): latest time to include in output

    Returns:
        none
    '''

    # Plot width and height in inches
    width, height = 10, 5
    pylab.rcParams['figure.figsize'] = width, height
    pylab.rcParams['figure.dpi'] = 70 # for export
    pylab.rcParams.update({'font.size': 22})
    pylab.rcParams.update({'legend.fontsize': 11.8})

    # Space above plot
    print('\n\n')

    # Uncertainties (sigma and 2sigma)
    y1 = np.array([mu[i] + 1 * np.sqrt(sigma[i][i]) for i in range(len(Xs))]).ravel()
    y2 = np.array([mu[i] - 1 * np.sqrt(sigma[i][i]) for i in range(len(Xs))]).ravel()
    pylab.fill_between(Xs.ravel(), y1, y2, linewidth=0, color='orange', alpha=0.5, label='1 SD')
    y1 = np.array([mu[i] + 2 * np.sqrt(sigma[i][i]) for i in range(len(Xs))]).ravel()
    y2 = np.array([mu[i] - 2 * np.sqrt(sigma[i][i]) for i in range(len(Xs))]).ravel()
    pylab.fill_between(Xs.ravel(), y1, y2, linewidth=0, color='orange', alpha=0.2, label='2 SD')

    # measurements and truth data
    pylab.plot(ttrue, ytrue, '.b', label='test data')
    pylab.plot(t, y, '.r', label='training data')

    # mean function
    pylab.plot(Xs, mu, '-g', alpha=0.5, label='mean prediction')

    # function draws
    draws = 0
    for i in range(draws):
        fs = getFunctionSample(mu, sigma)
        pylab.plot(Xs, fs, color='green', alpha=0.3, label='function draws')

    # Axis limits
    yl = 3
    pylab.ylim((-yl, yl))
    # pylab.xlim((4,5))

    ncol = 5

    # Time cutoff
    if time_cutoff:
        pylab.plot(np.array([time_cutoff, time_cutoff])*t[-1], [-yl, yl], '--k', label='cutoff')
        ncol = 3


    # Labels
    pylab.legend(loc='upper center', ncol=ncol)
    pylab.xlabel('time (days)')
    pylab.ylabel('tide height (normalised)')
    if title: pylab.title(title)

    pylab.show()