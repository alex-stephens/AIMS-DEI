import numpy as np
import pandas as pd

import dateutil.parser as dp


def loadData():
    '''
    Load data from online and store into arrays.

    Arguments:
        none

    Returns:
        t (np.array): time values for training data
        y (np.array): tide height values for training data
        ttrue (np.array): time values for test data
        ytrue (np.array): tide height values for test data
    '''

    data_url = 'http://www.robots.ox.ac.uk/~mosb/teaching/AIMS_CDT/sotonmet.txt'
    data = pd.read_csv(data_url)

    print(data.columns)

    # Time in days since first reading
    t = data['Reading Date and Time (ISO)']
    t = [int(dp.parse(x).strftime('%s')) for x in t]
    t = np.array([float(x - t[0])/86400 for x in t]).reshape(-1,1)

    # Training and test data
    y = data['Tide height (m)']
    ytrue = data['True tide height (m)']

    # Remove missing data points
    data_train = [(t[i], y[i]) for i in range(len(t)) if not np.isnan(y[i])]
    data_test = [(t[i], ytrue[i]) for i in range(len(t)) if not np.isnan(ytrue[i])]
    t, y = [x[0] for x in data_train], [x[1] for x in data_train]
    ttrue, ytrue = np.array([x[0] for x in data_test]).reshape(-1,1), [x[1] for x in data_test]

    # Normalise both datasetts
    mean, stdev = np.mean(y), np.std(y)
    y = np.array([(yi - mean) / stdev for yi in y]).reshape(-1,1)
    ytrue = np.array([(yi - mean) / stdev for yi in ytrue]).reshape(-1,1)

    return t, y, ttrue, ytrue