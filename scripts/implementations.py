import numpy as np
import threading
import random

############### utils function ###############

def process_data(x):
    """
    process_data processes the data as follows:
    - split the dataset in 3 datasets according to the jet number feature (for value 0, 1 and 2/3)
    - remove unnecessary features, i.e., highly correlated features, phi-features and undefined features with the jet number
    - set the remaining undefined values to the median of the feature
    - standardize the data

    Parameters:
        x (numpy.array): NxD matrix containing the dataset

    Returns:
        xs, idx_split (tuple(list(np.array))): the lists contain the processed data for each category and their respective index.
    """

    # indexes that we want to keep after the processing for each category
    idx_remaining_features = [
        [0,1,2,3,7,8,9,11,13,14,16,17,19,21],
        [0,1,2,3,7,8,9,11,13,14,16,17,19,21,23,24,29],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,19,21,23,24,26,27,28,29]
    ]

    idx_split = [
        np.argwhere(x[:,22]==0),
        np.argwhere(x[:,22]==1),
        np.argwhere(x[:,22]>1)
    ]

    xs = []

    for i in range(len(idx_remaining_features)):
        x_temp = x[idx_split[i], idx_remaining_features[i]]
        x_temp = set_undefined_to_median(x_temp)
        x_temp,_,_ = standardize(x_temp)
        xs.append(x_temp)

    return xs, idx_split

def set_undefined_to_median(x):
    """
    set_undefined_to_median sets the undefined values to the median of feature
    """
    medians = np.median(x, axis=0)
    for i in range(len(medians)):
        x[x[:,i] == -999, i] = medians[i]
    
    return x
    
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def compute_mse(y, tx, w):
    """
    Calculate the mse loss.
    """
    e = y - np.matmul(tx, w)
    N = len(y)
    return 1/(2*N)*e.T@e

def compute_rmse(y, tx, w):
    """Calculate the rmse loss.
    """
    return np.sqrt(2*compute_mse(y, tx, w))

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = tx.shape[0]
    D = tx.shape[1]
    I = np.eye(D)
    w = np.linalg.solve(tx.T@tx + 2*N*lambda_*I , tx.T@y)
    return compute_mse(y, tx, w), w

def get_accuracy(y, x, w):
    """"
    Compute the accuracy of a linear model compared to the labels
    """
    pred = x@w

    pred[pred < 0] = -1
    pred[pred > 0] = 1

    return np.count_nonzero(pred == y)/len(y)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N = x.shape[0]
    poly = np.ones((N,1))
    poly = np.c_[poly, np.sin(x)]
    poly = np.c_[poly, np.cos(x)]
    
    assert degree > 0
    
    if degree == 1:
        return np.c_[poly, x]

    for i in range(1, degree+1):
        poly = np.c_[poly, x**i]
    
    return poly

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

def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def split_data_from_k_indices(y, x, k_indices, k):
    te_idx = k_indices[k]
    tr_idx = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_idx = tr_idx.reshape(-1)
    
    y_te = y[te_idx]
    x_te = x[te_idx]
    
    y_tr = y[tr_idx]
    x_tr = x[tr_idx]

    return x_tr, y_tr, x_te, y_te

############### Class for our model ###############
class ThreeModels:
    """
    ThreeModels is the class for our model. It is composed of 3 underlying simpler (Ridge Regression) models.
    """
    def __init__(self, xs, ys):

        assert len(xs) == len(ys)

        self.models = [RidgeRegressionModel(xs[i], ys[i], f"model{i}") for i in range(len(xs))]
    
    def train(self, degrees, lambdas, k_folds):
        """
        train trains the 3 underlying models using CV.
        """
        threads = []

        # traine each model in its own threas
        for model in self.models:
            t = threading.Thread(target=model.CV_ridge_regression, args=(degrees, lambdas, k_folds))
            threads.append(t)
            print(f"Start training for model {model.name}")
            t.start()

        # wait for each thread to finish
        for t in threads:
            t.join()
        
        print(50*"=")
        print("Finish training")
        print(50*"=")
    
    def save_CV_results(self, path):
        """
        save_CV_results saves the results in a binary file at path containing (in this order) the np.arrays accuracy, rmse of training and rmse of testing
        """
        for model in self.models:
            assert model.rmse_tr is not None and model.rmse_te is not None and model.accuracy is not None
        
        for model in self.models:
            with open(f"{path}_{model.name}", "wb") as f:
                np.save(f, model.accuracy)
                np.save(f, model.rmse_tr)
                np.save(f, model.rmse_te)
    
    def load_CV_results(self, path):
        """
        load_CV_results loads the results from a file at path and populate the underlying np.array for accuracy, rmse of training and rmse of testing
        """
        for model in self.models:
            assert model.rmse_tr is None and model.rmse_te is None and model.accuracy is None
        
        for model in self.models:
            with open(f"{path}_{model.name}", "rb") as f:
                model.accuracy = np.load(f)
                model.rmse_tr = np.load(f)
                model.rmse_te = np.load(f)
    
    def compute_ws(self):
        for model in self.models:
            assert model.best_deg is not None and model.best_lambda is not None
        
        for model in self.models:
            x_poly = build_poly(model.x, model.best_deg)
            _, w = ridge_regression(model.y, x_poly, model.best_lambda)
            model.w = w

        


class RidgeRegressionModel:
    """
    RidgeRegressionModel is the class that contains the code for ridge regression
    """
    def __init__(self,x,y,name):
        self.x = x
        self.y = y
        self.rmse_te = None
        self.accuracy = None
        self.rmse_tr = None
        self.best_deg = None
        self.best_lambda = None
        self.name = name
        self.w = None


    def CV_ridge_regression(self,degrees, lambdas, k_folds):
        """
        CV_ridge_regression performs CV grid search for the hyper parameters degree and lambda
        """
        #prepare cross validation
        k_indices = build_k_indices(self.y, k_folds)

        #store loss in matrices
        rmse_tr = np.empty((len(degrees), len(lambdas)))
        rmse_te = np.empty((len(degrees), len(lambdas)))
        accuracy = np.empty((len(degrees), len(lambdas)))

        best_acc = 0
        best_deg = 0
        best_l = 0
        total_iter = len(degrees)*len(lambdas)
        it = 0
        steps = [0.1*i for i in range(1,10)]

        for idx_deg, d in enumerate(degrees):
            for idx_lambda, l in enumerate(lambdas):
                
                # print each 10%

                if len(steps) > 0 and it/total_iter > steps[0]:
                    steps = steps[1:]
                    print(50*"=")
                    print(self.name)
                    print("{:.2f}% of training done".format(100*it/total_iter))
                    print("Intermediate result:")
                    print("Degree: {:2d} Lambda: {:.10f} Accuracy: {:.3f}".format(best_deg, best_l, best_acc))
                    
                it += 1

                rmse_tr_tmp = []
                rmse_te_tmp = []
                accuracy_tmp = []
                for k in range(k_folds):
                    
                    # get the k-fold
                    x_tr, y_tr, x_te, y_te = split_data_from_k_indices(self.y, self.x, k_indices, k)
    
                    # build polynomial extension
                    x_te_poly = build_poly(x_te, d)
                    x_tr_poly = build_poly(x_tr, d)
                    
                    # train
                    _, w = ridge_regression(y_tr, x_tr_poly, l)
                    
                    # compute loss for training, loss and accuarcy for testing
                    rmse_tr_tmp.append(compute_rmse(y_tr, x_tr_poly, w))
                    rmse_te_tmp.append(compute_rmse(y_te, x_te_poly, w))
                    accuracy_tmp.append(get_accuracy(y_te, x_te_poly, w))
                
                rmse_te[idx_deg, idx_lambda] = np.mean(rmse_te_tmp)
                rmse_tr[idx_deg, idx_lambda] = np.mean(rmse_tr_tmp)
                mean_acc = np.mean(accuracy_tmp)
                accuracy[idx_deg, idx_lambda] = mean_acc

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_deg = d
                    best_l = l

        
        self.best_deg = best_deg
        self.best_lambda = best_l

        print(50*"=")
        print(self.name)
        print("Final result:")
        print("Degree: {:2d} Lambda: {:.10f} Accuracy: {:.3f}".format(best_deg, best_l, best_acc))

        self.rmse_te = rmse_te
        self.accuracy = accuracy
        self.rmse_tr = rmse_tr

############### mandatory function ###############

# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """"
    least_squares_GD implements linear regression using gradient descent.
    """
    w = initial_w
    for _ in range(max_iters):
        gradL = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradL
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """"
    least_squares_SGD implements linear regression using stochastic gradient descent.
    """
    w = initial_w
    for _ in range(max_iters):
        sample_idx = random.randint(0, len(y))
        gradL = compute_gradient(y[sample_idx], tx[sample_idx], w)
        loss = compute_mse(y[sample_idx], tx[sample_idx], w)
        w = w - gamma * gradL
    return w, loss


def least_squares(y, tx):
    """"
    least_squares implements least squares regression using normal equations
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, w)
    return w, loss

def compute_gradient(y, tx, w):
    """
    compute_gradient computes the gradient of a linear model for least squares.
    """
    e = y - tx @ w.T
    grad = -(1/len(y))* tx.T @ e
    return grad

def sigmoid(t):
    """
    sigmoid computes the sigmoid function
    """
    return np.exp(t)/(1+np.exp(t))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    """
    w = initial_w
    for _ in range(max_iters):
        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        w -= gamma*grad
    return w, loss

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sig = sigmoid(tx@w)
    loss = y.T@np.log(sig) + (1-y.T)@np.log(1-sig)
    return -loss[0,0]

def calculate_gradient(y, tx, w):
    """compute the gradient of loss for logistic regression."""
    return tx.T@(sigmoid(tx@w)-y)

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    """
    Regularized  logistic  regression  using  gradient  descent
    """
    w = initial_w

    for _ in range(max_iters):
        loss = calculate_loss(y, tx, w) + 0.5*lambda_*w.T@w
        grad = calculate_gradient(y, tx, w) + lambda_*w
        w -= gamma*grad
    return w, loss