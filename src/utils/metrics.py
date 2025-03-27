import numpy as np
from matplotlib import pyplot as plt

def my_accuracy_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_values = y_true == y_pred
    score = np.sum(true_values) / y_true.size
    return score

def my_precision_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    score = TP / (TP + FP)
    return score

def my_recall_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    score = TP / (TP + FN)
    return score

def my_f1_score(y_true, y_pred, beta=1.0):
    precision = my_precision_score(y_true, y_pred)
    recall = my_recall_score(y_true, y_pred)
    score = (1 + beta**2) * precision * recall / ( beta**2 * precision + recall )
    return score

def MSE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    score = np.sum((y_true - y_pred)**2) * 1/y_true.size
    return score

def RMSE(y_true, y_pred):
    mse = MSE(y_true, y_pred)
    score = np.sqrt(mse)
    return score

def MAE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    score = np.sum(np.abs(y_true - y_pred)) * 1/y_true.size
    return score

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    score = np.sum( np.abs(y_true - y_pred) / y_true ) * 1 / y_true.size
    return score

def SMAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    score = np.sum( np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) ) / y_true.size
    return score

def R_squared(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    SSR = np.sum( (y_true - y_pred)**2 )
    SST = np.sum( (y_true - np.mean(y_true))**2 )
    score = 1 - SSR / SST
    return score

def ROC_AUC(y_true, pred_proba):

    y_true, pred_proba = np.array(y_true), np.array(pred_proba)
    FPR, TPR = [], []
    thresholds = np.unique(pred_proba)
    P = np.sum(y_true)
    N = y_true.size - P
    # iterate thresholds
    for thresh in thresholds:
        FP, TP = 0, 0
        thresh = np.round(thresh, 2)
        for i in range(pred_proba.size):
            if pred_proba[i] >= thresh:
                if y_true[i] == 1:
                    TP += 1
                if y_true[i] == 0:
                    FP += 1
        FPR.append(FP/N)
        TPR.append(TP/P)

    auc = -1 * np.trapezoid(TPR, FPR)

    '''
    plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw=2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.2f' % auc)
    plt.legend(loc="lower right")
    plt.savefig('AUC_example.png')
    plt.show()
    '''

    return auc








