import pylab
import numpy as np

def getFunctionSample(mu, sigma):
    fs = np.random.multivariate_normal(mu, sigma, 1)
    return np.transpose(fs)


def plotTides(Xs, mu, sigma, t, y, ttrue, ytrue):

    # Plot width and height in inches
    width, height = 10, 5
    pylab.rcParams['figure.figsize'] = width, height
    pylab.rcParams['figure.dpi'] = 100 # for export
    pylab.rcParams.update({'font.size': 22})
    pylab.rcParams.update({'legend.fontsize': 12})

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
    pylab.plot(t, y, '.r', label='training data')
    pylab.plot(ttrue, ytrue, '.b', label='test data')

    # mean function
    pylab.plot(Xs, mu, '-r', alpha=0.5, label='mean function')

    # function draws
    draws = 0
    for i in range(draws):
        fs = getFunctionSample(mu, sigma)
        pylab.plot(Xs, fs, color='green', alpha=0.3, label='function draws')

    pylab.legend(loc='upper center', ncol=5)
    pylab.xlabel('time (days)')
    pylab.ylabel('tide height (normalised)')
    yl = 3
    pylab.ylim((-yl, yl))

    pylab.show()