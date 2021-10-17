# -*- coding: utf-8 -*-
import numpy as np
"""Function used to compute the loss."""

def compute_mse(y, tx, w):
    """
    Calculate the mse loss.
    """
    e = y - np.matmul(tx, w)
    N = len(y)
    return 1/(2*N)*np.dot(e, e)

def compute_rmse(y, tx, w):
    """Calculate the rmse loss.
    """
    return np.sqrt(2*compute_mse(y, tx, w))

def compute_mae_loss(y, tx, w):
    """Calculate the mae loss.
    """
    e = y - np.matmul(tx, w)
    N = len(y)
    return np.mean(e)/N

def get_accuracy(y, x, w):
    """"
    Compute the accuracy of a linear model compared to the labels
    """
    pred = x@w

    pred[pred < 0] = -1
    pred[pred > 0] = 1

    return np.count_nonzero(pred == y)/len(y)