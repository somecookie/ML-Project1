import numpy as np

def remove999(x):
    toRemove=np.zeros(len(x.T))
    for i in range(len(x.T)):
        toRemove[i] = ((-999.) in x[:,i])*i
    toRemove = toRemove[toRemove !=0].astype(int)
    xClean = np.delete(x, toRemove, 1)
    return xClean

def compute_loss(y, tx, w):
    e = y - tx @ w
    mse = len(y)**(-1) * e.T @ e
    return mse

def compute_loss2(y, tx, w):
    e = y - tx @ w
    mae = len(y)**(-1) * np.sum(np.absolute(e))
    return mae

def calculate_logloss(y, tx, w):
    loss = 0
    for i in range(len(y)):
        loss = loss + np.log(1+np.exp(tx[i]@w))-y[i]*tx[i]@w
    return loss

def clean_data(x):
    mean_x = np.nanmean(x, axis=0)
    median_x = np.nanmedian(x, axis=0)
    x = x - mean_x
    std_x = np.nanstd(x, axis=0)
    x = x / std_x
    return x, mean_x, median_x

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    return w

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    xPoly = np.zeros((len(x), degree+1))
    for i in range(degree+1):
        xPoly[:, i] = x**i
    return xPoly

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    w = np.linalg.solve(tx.T @ tx + 2 * len(tx) * lambda_ * np.eye(len(tx.T)), tx.T @ y)
    return w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for iter in range(max_iters):
        print('sigm: ', sigmoid(tx@w))
        gradL = tx.T@(sigmoid(tx@w)-y)
        #gamma = 0.001*np.linalg.norm(gradL)
        w = w - gamma * gradL
        loss = calculate_logloss(y, tx, w)
        print('Loss at time ', iter, ' is ', loss)
    return w, loss

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    w = initial_w
    for iter in range(max_iters):
        #print('sigm: ', sigmoid(tx@w))
        #print('w: ', w.T)
        gradL = tx.T@(sigmoid(tx@w)-y)+lambda_*w
        print('grad: ', gradL.T)
        #gamma = 1e-8*np.linalg.norm(gradL)
        w = w - gamma * gradL
        loss = calculate_logloss(y, tx, w)
        print('Loss at time ', iter, ' is ', loss)
    return w, loss

def logistic_reg_newton(y, tx, initial_w, max_iters, gamma, lambda_):
    w = initial_w
    for iter in range(max_iters):
        print('Calculating diagonal... Be patient (', iter+1, '/', max_iters, ').')
        diagS = sigmoid(tx@w)*(np.ones((len(y),1))-sigmoid(tx@w))
        print('Calculating S... Be patient (', iter+1, '/', max_iters, ').')
        S = np.diagflat(diagS[:])
        print('Calculating H... Be patient (', iter+1, '/', max_iters, ').')
        H = tx.T@S@tx
        gradL = tx.T@(sigmoid(tx@w)-y)
        print('Inverting... Be patient (', iter+1, '/', max_iters, ').')
        w = np.linalg.solve(H, H@w-gamma*gradL)
        loss = calculate_logloss(y, tx, w)
        print('Loss at time ', iter, ' is ', loss)
    return w, loss

def split_data(x, y, ratio, seed=1):
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

def get_accuracy(yGuess, yTrue):
    trueGuess = len(yTrue)-np.count_nonzero(np.abs(np.sign(yGuess)-yTrue))
    accuracy = trueGuess/len(yTrue)
    return accuracy

def get_accuracyLog(yGuess, yTrue):
    guess1 = (yGuess-0.5)>0
    trueGuess = len(yTrue)-np.count_nonzero(guess1-yTrue)
    accuracy = trueGuess/len(yTrue)
    return accuracy

def sigmoid(t):
    sigm = 1/(np.ones(t.shape)+np.exp(-t))
    return sigm

def split_cat(x, y): #removed 15, 18, 20, 25
    cat1Id = np.argwhere(x[:,22]==0)
    cat2Id = np.argwhere(x[:,22]==1)
    cat3Id = np.argwhere(x[:,22]>1)
    cat1Idx = [0,1,2,3,7,8,9,10,11,13,14,16,17,19,21]
    cat2Idx = [0,1,2,3,7,8,9,10,11,13,14,16,17,19,21,23,24,29]
    cat3Idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,19,21,23,24,26,27,28,29]
    cat1X = x[cat1Id, cat1Idx]
    cat2X = x[cat2Id, cat2Idx]
    cat3X = x[cat3Id, cat3Idx]
    cat1Y = y[cat1Id]
    cat2Y = y[cat2Id]
    cat3Y = y[cat3Id]
    return cat1X, cat2X, cat3X, cat1Y, cat2Y, cat3Y

def split_cat_web(x):
    cat1Id = np.argwhere(x[:,22]==0)
    cat2Id = np.argwhere(x[:,22]==1)
    cat3Id = np.argwhere(x[:,22]>1)
    cat1Idx = [0,1,2,3,7,8,9,10,11,13,14,16,17,19,21]
    cat2Idx = [0,1,2,3,7,8,9,10,11,13,14,16,17,19,21,23,24,29]
    cat3Idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,19,21,23,24,26,27,28,29]
    cat1X = x[cat1Id, cat1Idx]
    cat2X = x[cat2Id, cat2Idx]
    cat3X = x[cat3Id, cat3Idx]
    return cat1X, cat2X, cat3X, cat1Id, cat2Id, cat3Id

def build_features_logistic(x, y):
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return tx
