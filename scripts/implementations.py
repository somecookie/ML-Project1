import numpy as np
import matplotlib as plt
import threading

def split_data_jet_num(x, y, headers):
    """
    split_data_jet_num splits the data in 3 according to the jet number value (0, 1 and 2-3).
    
    Parameters:

        x (numpy.array)     : NxD matrix containing the dataset
        headers (list(str)) : list of length D containing the names of the features
    
    Returns:
        x0, x1, x23 (tuple((numpy.array))): 3 datasets that for which the value PRI_jet_num is 0, 1 or 2-3
    """

    jet_num_idx = headers.index('PRI_jet_num')
    x0 = x[x[:, jet_num_idx] == 0, :]
    x0 = np.concatenate((x0[:, :jet_num_idx], x0[:, jet_num_idx+1:]), axis=1)
    y0 = y[x[:, jet_num_idx] == 0]
    x1 = x[x[:, jet_num_idx] == 1, :]
    x1 = np.concatenate((x1[:, :jet_num_idx], x1[:, jet_num_idx+1:]), axis=1)
    y1 = y[x[:, jet_num_idx] == 1]
    x2 = x[x[:, jet_num_idx] == 2, :]
    y2 = y[x[:, jet_num_idx] == 2]
    x3 = x[x[:, jet_num_idx] == 3, :]
    y3 = y[x[:, jet_num_idx] == 3]
    x23 = np.concatenate((x2,x3))
    x23 = np.concatenate((x23[:, :jet_num_idx], x23[:, jet_num_idx+1:]), axis=1)
    y23 = np.concatenate((y2,y3))

    return x0, y0, x1, y1, x23,y23

def remove_undefined_features(x, headers, print_undefined=False):
    """
    remove_undefined_features removes the undefined features, i.e., the features for which all entries are -999.

    Parameters:

        x (numpy.array)         : NxD matrix containing the dataset
        headers (list(str))     : list of length D containing the names of the features
        print_undefined (bool)  : print the undefined headers in alphabetical order
    
    Returns:
        numpy.array: the dataset without the undefined features
    """

    defined_idx = np.mean(x, axis=0) > -999

    if print_undefined:
        undefined_headers = "\n".join(sorted(["- " + headers[i] for i in np.arange(x.shape[1])[~defined_idx]]))
        print(f"Undefined headers:\n{undefined_headers}")
    
    return x[:, defined_idx]

def set_undefined_to_median(x):
    """
    set_undefined_to_median sets the undefined values to the median of feature
    """
    medians = np.median(x, axis=0)
    for i in range(len(medians)):
        x[x[:,i] == -999, i] = medians[i]
    
    return x

def remove_phi_features(x, headers):
    """
    remove_phi_features removes the features ending with the suffix _phi
    """
    idx = [i for i in range(len(headers)) if headers[i].endswith('_phi')]
    new_size = len(idx)
    new_x = np.zeros((x.shape[0], x.shape[1]-new_size))
    new_headers = []
    j = 0

    for i in range(len(headers)):
        if i in idx:
            continue

        new_headers.append(headers[i])
        new_x[:, j] = x[:, i]
        j+=1

    return new_x, new_headers

def find_correlated_features(x, threshold=0.7):
    """
    find_correlated_features finds all pairs of features which have a correlation higher than the threshold.
    """

    correlated_features = {}

    for i in range(x.shape[1]-1):
        for j in range(i+1, x.shape[1]):
            xi = x[:, i]
            xj = x[:, j]

            corr = np.corrcoef(xi, xj)[0,1]

            if corr > threshold:
                correlated_features[i] = j

    return correlated_features
    
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
    return 1/(2*N)*np.dot(e, e)

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
    assert degree > 0
    
    if degree == 1:
        return np.c_[poly, x]

    for i in range(1, degree+1):
        poly = np.c_[poly, x**i]
    
    return poly

class ThreeModels:
    def __init__(self, x0,y0,x1,y1,x23,y23):
        self.models = [RidgeRegressionModel(x0, y0, "model0"), RidgeRegressionModel(x1, y1, "model1"), RidgeRegressionModel(x23, y23, "model23")]
    
    def train(self, degrees, lambdas, k_folds):
        threads = []

        # traine each model in its own threas
        for model in self.models:
            t = threading.Thread(target=model.CV_ridge_regression, args=(degrees, lambdas, k_folds))
            threads.append(t)
            t.start()

        # wait for each thread to finish
        for t in threads:
            t.join()
        
        print("Finish training")

class RidgeRegressionModel:
    def __init__(self,x,y,name):
        self.x = x
        self.y = y
        self.rmse_te = None
        self.accuracy = None
        self.rmse_tr = None
        self.best_deg = None
        self.best_lambda = None
        self.name = name


    def CV_ridge_regression(self,degrees, lambdas, k_folds, print_res=False):
        #prepare cross validation
        k_indices = build_k_indices(self.y, k_folds)

        #store loss in matrices
        rmse_tr = np.empty((len(degrees), len(lambdas)))
        rmse_te = np.empty((len(degrees), len(lambdas)))
        accuracy = np.empty((len(degrees), len(lambdas)))

        for idx_deg, d in enumerate(degrees):
            for idx_lambda, l in enumerate(lambdas):
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
                
                rmse_te_mean = np.mean(rmse_te_tmp)
                rmse_tr_mean = np.mean(rmse_tr_tmp)
                accuracy_mean = np.mean(accuracy_tmp)
                
                if print_res:
                    print("D: {:2d} L: {:1.10f} RMSE TR: {:.5f} RMSE TE: {:.5f} ACC: {:1.5f}".format(d, l, rmse_tr_mean, rmse_te_mean,accuracy_mean))
                
                rmse_te[idx_deg, idx_lambda] = rmse_te_mean
                rmse_tr[idx_deg, idx_lambda] = rmse_tr_mean
                accuracy[idx_deg, idx_lambda] = accuracy_mean

        max_acc_idx = np.where(accuracy == np.max(accuracy))
        
        self.best_deg = degrees[max_acc_idx[0]][0]
        self.best_lambda = lambdas[max_acc_idx[1]][0]

        print("{:7s} Degree: {:2d} Lambda: {:.10f} Accuracy: {:.3f}, ".format(self.name,self.best_deg, self.best_lambda, np.max(accuracy)))

        self.rmse_te = rmse_te
        self.accuracy = accuracy
        self.rmse_tr = rmse_tr


def heatmap_accuracy(accuracy, lambdas, degrees, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(accuracy.T)


    # We want to show all ticks...
    ax.set_yticks(np.arange(len(lambdas)))
    ax.set_xticks(np.arange(len(degrees)))
    # ... and label them with the respective list entries
    ax.set_yticklabels(lambdas)
    ax.set_xticklabels(degrees)
    ax.set_ylabel("Lambda")
    ax.set_xlabel("Degree")

    fig.colorbar(im, ax=ax)

    ax.set_title("Accuracy for the ridge regression CV on the degree and lambda")

    fig.tight_layout()
    plt.show()

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

def get_rates(y, pred):
    # predict 1 and label is 1
    true_pos = np.sum(np.logical_and(pred == 1, y == 1))
    # predict -1 and label is -1
    true_neg = np.sum(np.logical_and(pred == -1, y == -1))
    # predict 1 and label is -1
    false_pos = np.sum(np.logical_and(pred == 1, y == -1))
    # predict -1 and label is 1
    false_neg = np.sum(np.logical_and(pred == -1, y == 1))

    return true_pos, true_neg, false_pos, false_neg

def get_accuracy_from_rates(true_pos, true_neg, false_pos, false_neg):
    return (true_pos + true_neg)/(true_pos + true_neg + false_neg + false_pos)

def get_f1score(true_pos, false_pos, false_neg):
    return true_pos/(true_pos + 0.5*(false_pos + false_neg))

def show_confusion_matrix(true_pos, true_neg, false_pos, false_neg):
    confusion_matrix = np.array([[true_pos, false_pos], [false_neg, true_neg]])

    _, ax= plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
