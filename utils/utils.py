import numpy as np
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer


def custom_softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stabilize softmax
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_target_vector(y: np.ndarray, n_class: int) -> np.ndarray:
    labelencoder_y = LabelBinarizer()
    return labelencoder_y.fit_transform(y)

def mapping_prob_to_class(y_logistic: np.ndarray) -> np.ndarray:
    """
    :param y_logistic: encoded labels.
    :return: list of class labels obtained from encoded labels.
    """
    return np.argmax(y_logistic, axis=1).tolist()

def accuracy(y_pred: np.ndarray, y_original:np.ndarray) -> float:
    """ Compute accuracy comparing matches between predicted and original class labels.
    :param y_pred: predicted class labels (list).
    :param y_original: original class labels (list).
    :return: accuracy percentage (float).
    """
    y_pred_arr = np.asarray(y_pred)
    y_original_arr = np.asarray(y_original)
    return float(np.mean(y_pred_arr == y_original_arr) * 100)