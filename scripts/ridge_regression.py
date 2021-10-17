# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
import costs

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    N = tx.shape[0]
    D = tx.shape[1]
    I = np.eye(D)
    w = np.linalg.solve(tx.T@tx + 2*N*lambda_*I , tx.T@y)
    return costs.compute_mse(y, tx, w), w
