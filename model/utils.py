import numpy as np
from sklearn.metrics import f1_score, log_loss


def f1_micro(y_true, y_pred):
    return -f1_score(y_true, y_pred, average="micro", zero_division=0)


def f1_micro_lgb(y_true, y_pred):
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return "f1_micro", f1_score(y_true, y_pred, average="micro", zero_division=0), True

def binary_logloss(y_true, y_pred):
    return "binary_logloss", log_loss(y_true, y_pred), False
