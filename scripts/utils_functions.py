# CS-433 - ML Project 1

import random
import numpy as np

# Base functions----------------------------------------------------------------------------------------------------

# Computes the gradient for a linear model
def compute_gradient(y, tx, w):
    e = y - tx @ w.T
    grad = -(1/len(y))* tx.T @ e
    return grad

# Computes the MSE for a linear model
def compute_loss_mse(y, tx, w):
    e = y - tx @ w
    mse = (1/len(y)) * e.T @ e
    return mse

# Computes the MAE for a linear model
def compute_loss_mae(y, tx, w):
    e = y - tx @ w
    mae = (1/len(y)) * np.sum(np.absolute(e))
    return mae

#Computes the sigmoid function
def sigmoid(t):
    return np.exp(t)/(1+np.exp(t))

#Computes the negative log likelyhood
def calculate_logloss(y, tx, w):
    loss = 0
    for i in range(len(y)):
        loss = loss + np.log(1+np.exp(tx[i].T@w))-y[i]*tx[i].T@w
    return loss


# Mandatory functions--------------------------------------------------------------------------------------------

# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradL = compute_gradient(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for iter in range(max_iters):
        gradL = tx.T@(sigmoid(tx@w)-y)
        w = w - gamma * gradL
        loss = calculate_logloss(y, tx, w)
    return w, loss

def learning_by_gradient_descent(y, tx, w, gamma):
    loss = calculate_loss(y, tx, w)
    gradL = calculate_gradient(y, tx, w)
    w = w-gamma*gradL
    return loss, w

# Other useful functions--------------------------------------------------------------------------------------------

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    
    idx_tr = idx[:int(n*ratio)]
    idx_te = idx[int(n*ratio):]
    
    train_x = x[idx_tr]
    train_y = y[idx_tr]
    test_x = x[idx_te]
    test_y = y[idx_te]
    
    return train_x, train_y, test_x, test_y

def least_square_training(x_tr, y_tr, x_te, y_te):
    """"
    Trains a linear model using the training data x_tr and y_tr and tests it against
    the testing data x_te and y_te.
    Returns the weight 
    """
    # Compute least square with explicit form
    w_ls, mse_ls_tr = least_squares(y_tr, x_tr)
    rmse_ls_tr = np.sqrt(2*mse_ls_tr)
    mse_ls_te = compute_loss_mse(y_te, x_te, w_ls)
    rmse_ls_te = np.sqrt(2*mse_ls_te)

    # Prediction on test set
    pred_te = x_te @ w_ls
    pred_te[pred_te < 0] = -1
    pred_te[pred_te > 0] = 1

    accuarcy = np.count_nonzero(pred_te == y_te)/len(pred_te)

    print(f"Training RMSE: {rmse_ls_tr} Testing RMSE: {rmse_ls_te} with accuarcy: {accuarcy}")
    return w_ls

def set_NaN_to_median(x_tr, x_te):
    medians = np.median(x_tr, axis=0)
    medians_x_tr = np.copy(x_tr)

    for col_idx, median in enumerate(medians):
        idx = medians_x_tr[:, col_idx] == -999
        medians_x_tr[idx, col_idx] = median

    medians = np.median(x_te, axis=0)
    medians_x_te = np.copy(x_te)

    for col_idx, median in enumerate(medians):
        idx = medians_x_te[:, col_idx] == -999
        medians_x_te[idx, col_idx] = median
    
    return (medians_x_tr, medians_x_te)

def zscore_normalize(x_tr, x_te):
    norm_x_tr = np.copy(x_tr)
    mean_tr = np.reshape(np.mean(norm_x_tr, axis=0), (1,-1))
    std_tr = np.reshape(np.std(norm_x_tr, axis=0), (1,-1))

    norm_x_tr = (norm_x_tr - mean_tr)/std_tr

    norm_x_te = np.copy(x_te)
    mean_te = np.reshape(np.mean(norm_x_te, axis=0), (1,-1))
    std_te = np.reshape(np.std(norm_x_te, axis=0), (1,-1))

    norm_x_te = (norm_x_te - mean_te)/std_te

    return (norm_x_tr, norm_x_te)

