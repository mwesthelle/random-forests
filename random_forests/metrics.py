import numpy as np


def precision(y_predictions, y_values) -> float:
    TP = np.sum(np.logical_and(y_predictions == 1, y_values == 1))
    FP = np.sum(np.logical_and(y_predictions == 1, y_values == 0))
    return TP / (TP + FP)


def recall(y_predictions, y_values) -> float:
    TP = np.sum(np.logical_and(y_predictions == 1, y_values == 1))
    FN = np.sum(np.logical_and(y_predictions == 0, y_values == 1))
    return TP / (TP + FN)


def f_measure(y_predictions, y_values, beta=1) -> float:
    prec = precision(y_predictions, y_values)
    rec = recall(y_predictions, y_values)
    beta_squared = beta ** 2
    return (1 + beta_squared) * (prec * rec) / ((beta_squared * prec) + rec)
