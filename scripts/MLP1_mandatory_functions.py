# CS-433 - ML Project 1

import random
import numpy as np

# Base functions----------------------------------------------------------------------------------------------------

# Computes the gradient for a linear model
def compute_gradient(y, tx, w):
    e = y - tx @ w.T
    grad = -len(y) ** (-1) * tx.T @ e
    return grad

# Computes the MSE for a linear model
def compute_loss_mse(y, tx, w):
    e = y - tx @ w
    mse = len(y) ** (-1) * e.T @ e
    return mse

# Computes the MAE for a linear model
def compute_loss_mae(y, tx, w):
    e = y - tx @ w
    mae = len(y) ** (-1) * np.sum(np.absolute(e))
    return mae

# Mandatory functions--------------------------------------------------------------------------------------------

# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradL = compute_gradient(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        # gamma = np.linalg.norm(gradL)*0.1
        w = w - gamma * gradL
    return w, loss

#Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        sample = random.randrange(0, len(y))
        gradL = compute_gradient(y[sample], tx[sample], w)
        loss = compute_loss_mse(y[sample], tx[sample], w)
        w = w - gamma * gradL
    return w, loss

#Least squares regression using normal equations
def least_squares(y, tx):
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss

#Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    N = len(tx.T)
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.eye(N), tx.T @ y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss