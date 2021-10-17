import numpy as np
import matplotlib.pyplot as plt

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

def get_accuracy(true_pos, true_neg, false_pos, false_neg):
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