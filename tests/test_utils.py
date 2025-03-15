import numpy as np
from utils.utils import (
    custom_softmax,
    compute_target_vector,
    mapping_prob_to_class,
    accuracy
)

def test_custom_softmax():
    z = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    softmax_result = custom_softmax(z)
    
    assert softmax_result.shape == z.shape
    assert np.allclose(np.sum(softmax_result, axis=1), np.ones(z.shape[0]))

def test_compute_target_vector():
    y = np.array([0, 1, 2, 1, 0])
    n_class = 3
    target_vector = compute_target_vector(y, n_class)
    
    assert target_vector.shape == (len(y), n_class)
    assert np.all((target_vector.sum(axis=1) == 1))  # Each row should have exactly one '1'

def test_mapping_prob_to_class():
    y_logistic = np.array([[0.1, 0.7, 0.2], [0.3, 0.2, 0.5], [0.8, 0.1, 0.1]])
    predicted_classes = mapping_prob_to_class(y_logistic)
    
    expected_classes = [1, 2, 0]
    assert predicted_classes == expected_classes

def test_accuracy():
    y_pred = np.array([0, 1, 2, 1, 0])
    y_original = np.array([0, 1, 1, 1, 0])
    acc = accuracy(y_pred, y_original)
    
    expected_accuracy = (4 / 5) * 100  # 4 out of 5 are correct
    assert acc == expected_accuracy
