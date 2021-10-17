import numpy as np

def zscore(x):
    """"
    This standardize the data in x using the zscore standardization method.
    The return value is (x-mean(x))/std(x)
    """
    means = np.mean(x, axis=0)
    stdev = np.std(x, axis=0)

    return (x-means)/stdev