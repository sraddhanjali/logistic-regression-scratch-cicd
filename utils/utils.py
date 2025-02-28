import numpy as np
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer

def custom_softmax(unsoftmaxed_data: np.ndarray, axis: int =0) -> np.ndarray:
    def softmax_func(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    return np.apply_along_axis(softmax_func, axis, unsoftmaxed_data)

def compute_target_vector(y_original: np.ndarray, n_class: int) -> np.ndarray:
    if n_class == 2:
        y_encoded = np.zeros(len(y_original), 2)
        for i, label in enumerate(y_original):
            y_encoded[i, int(label)] = 1
    else:
        y_encoded = LabelBinarizer().fit_transform(y_original)
    return y_encoded

def mapping_three_prob_to_class(y_logistic: np.ndarray) -> np.ndarray:
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