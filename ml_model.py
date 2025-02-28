import numpy as np
from utils.utils import custom_softmax, compute_target_vector

def gradient_mat(phi_matrix, old_weight, n_class, target_vec, reg_param_lambda):
    y = phi_matrix @ old_weight  # Matrix multiplication
    y = custom_softmax(y, axis=1)
    reg_term = (reg_param_lambda / (2 * len(y))) * np.sum(np.square(old_weight))
    cost = -np.sum(target_vec * np.log(y)) / len(y) + reg_term
    grad_matrix = (phi_matrix.T @ (y - target_vec)) / len(y) + reg_param_lambda * old_weight
    return grad_matrix, cost

def grad_descent(phi_matrix, n_class, curr_weight, l_rate, max_iter, target_vec, reg_param_lambda):
    old_weight = np.zeros_like(curr_weight)
    cost_history = []
    for _ in range(max_iter):
        if np.allclose(old_weight, curr_weight): break
        old_weight = curr_weight.copy()
        grad_matrix, cost = gradient_mat(phi_matrix, old_weight, n_class, target_vec, reg_param_lambda)
        curr_weight -= l_rate * grad_matrix
        cost_history.append(cost)
    return curr_weight, cost_history
