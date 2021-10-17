import numpy as np
import matplotlib.pyplot as plt
from split_data import split_data_from_k_indices, build_k_indices
from build_polynomial import build_poly
from costs import compute_rmse, get_accuracy
from ridge_regression import ridge_regression

def CV_ridge_regression(x, y, degrees, lambdas, k_folds, print_res=True):
    #prepare cross validation
    k_indices = build_k_indices(y, k_folds)

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
                x_tr, y_tr, x_te, y_te = split_data_from_k_indices(y, x, k_indices, k)
                
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
    max_deg = degrees[max_acc_idx[0]][0]
    max_lambda = lambdas[max_acc_idx[1]][0]

    if print_res:
        print("Degree: {:2d} Lambda: {:.10f} Accuracy: {:.3f}, ".format(max_deg, max_lambda, np.max(accuracy)))

    return rmse_te, accuracy, rmse_tr

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
