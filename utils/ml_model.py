import numpy as np
import sys, os
from utils.utils import custom_softmax, compute_target_vector, mapping_prob_to_class
from utils.config import CONFIG
from utils.preprocessing import ScalerTransform
from utils.features import PhiMatrixTransformer, TransformerMixin
from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from utils.config import CONFIG


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


class GradientDescentOptimizer:
    def __init__(self) -> None:
        self.l_rate = float(CONFIG["l_rate"])
        self.max_iteration = float(CONFIG["max_iteration"])
        self.reg_param_lambda = float(CONFIG["reg_param_lambda"])
        self.cost_history = None
        self.cost_iterations = None
        self.grad_matrix_history = None
        self.phi_matrix = None
        self.curr_weight: Optional[np.ndarray] = None

    def fit(self, X: np.matrix, y: np.ndarray) -> 'GradientDescentOptimizer':
        # print("Gradient Descent fit got called", y.shape)
        _n_class = int(CONFIG["n_class"])
        target_vec = compute_target_vector(y, n_class=_n_class)
        self.curr_weight = np.random.random_sample((X.shape[1], _n_class))
        self.curr_weight, self.cost_history, self.cost_iterations, self.grad_matrix_history = grad_descent(
            X, self.curr_weight, self.l_rate, self.max_iteration, target_vec, self.reg_param_lambda, n_class=_n_class)
        self.is_fitted_ = True
        # print("this fit is being called and weights are ", len(self.curr_weight))
        return self
    

class LogisticRegressionFromScratch(BaseEstimator, TransformerMixin):

    
    def __init__(self) -> None:
        # maintain state of learnable parameters over training etc.
        self.l_rate = CONFIG["l_rate"]
        self.max_iteration = CONFIG["max_iteration"]
        self.reg_param_lambda = CONFIG["reg_param_lambda"]
        self.weights = []
        self.cost_history = []
        self.cost_iterations = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionFromScratch':
        # print("fit got called", X.shape, y.shape)
        self.pipeline.fit(X, y)
        self.cost_history = self.pipeline.named_steps['gradient_descent'].cost_history
        self.cost_iterations = self.pipeline.named_steps['gradient_descent'].cost_iterations
        self.weights = self.pipeline.named_steps['gradient_descent'].curr_weight
        return self
    
    def summary(self) -> str:
        return "summary"

    def predict(self, X: np.ndarray) -> np.ndarray:
        print("predict got called", X.shape)
        print("weights, ", len(self.weights))
        y_log_t = (self.weights * X).T
        print(y_log_t.shape)
        y_logistic = custom_softmax(y_log_t, along_axis=1)
        print(y_logistic.shape)
        mapped_y_pred = mapping_prob_to_class(y_logistic)
        return mapped_y_pred

    def plot_convergence(self) -> None:
        cost_history = list(map(lambda x: x.to_list()[0], self.cost_iterations))
        # for i in range(len(self.cost_iterations)):
            # self.cost_history[i] = self.cost_history[i].tolist()
        # self.cost_history = [i[0] for i in self.cost_history]
        plt.plot(self.cost_iterations, cost_history)
        plt.title('Cost Function')
        plt.xlabel('Iterations')
        plt.ylabel('J_cost')
        plt.show()

    def save_checkpoint(self, filename: str) -> None:
        curr_dir = sys.path('.')
        file_path = os.path.join(curr_dir, "models", filename)
        np.savez(file_path, weights=self.pipeline.named_steps['gradient_descent'].curr_weight)
