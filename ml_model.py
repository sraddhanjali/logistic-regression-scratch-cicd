import numpy as np
import os
from utils.utils import custom_softmax, compute_target_vector, mapping_prob_to_class, accuracy
from utils.config import CONFIG
from utils.preprocessing import ScalerTransform
from utils.features import PhiMatrixTransformer, TransformerMixin
from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted
from utils.config import CONFIG

import logging as l

from typing import List


logger = l.Logger('LogisticRegressionLogger')


class GradientDescentOptimizer:
    def __init__(self) -> None:
        self.l_rate: float = float(CONFIG["l_rate"])
        self.max_iteration: int = int(np.round(float(CONFIG["max_iteration"])))
        self.reg_param_lambda: float = float(CONFIG["reg_param_lambda"])
        self.cost_history: List = []
        self.grad_matrix: List = []
        self.cost_iterations: List = []
        self.curr_weight: np.ndarray =  None

    def gradient_mat(self, phi_matrix, old_weight, target_vec, reg_param_lambda, n_class):
        """ y = phi_matrix * old_weight
            n_samples = y.shape[0]
            y = custom_softmax(y, along_axis=1)
            b = (1/n_samples) * reg_param_lambda * np.sum(np.square(old_weight))
            tar_vector_sh = (target_vec.reshape((n_samples * n_class), 1)).T
            y_sh = y.reshape((n_samples*n_class), 1)
            cost_val = -(tar_vector_sh * np.log(y_sh))/n_samples + b
            diff = y - target_vec
            grad_matrix = ((phi_matrix.T * diff) + reg_param_lambda * old_weight) / n_samples
        """
        y = phi_matrix @ old_weight  # Matrix multiplication
        n_samples = y.shape[0]
        y_new = custom_softmax(y)
        # Ensure shapes match
        if y.shape != y_new.shape:
            logger.error(msg=str(f"shape mismatch: y shape {y.shape} and custom softmaxed y: {y_new}"))
            raise ValueError(f"Shape mismatch: y ")
        logger.info(msg=str(f"n_samples {n_samples} - n_class {n_class} - y shape {y.shape}"))
        y_sh = y_new.reshape((n_samples* n_class), 1)
        # Compute cross-entropy loss safely
        target_vec_sh = (target_vec.reshape((n_samples*n_class), 1)).T
        cost = -np.sum(target_vec_sh @ np.log(y_sh + 1e-8)) / n_samples
        
        # L2 Regularization term
        reg_term = (reg_param_lambda / (n_samples)) * np.sum(np.square(old_weight))
        cost += reg_term

        # Compute gradient
        delta = y - target_vec
        grad_matrix = ((phi_matrix.T @ delta ) + reg_param_lambda * old_weight) / n_samples
        return grad_matrix, cost

    def grad_descent(self, phi_matrix, l_rate, max_iter, target_vec, reg_param_lambda, n_class):
        self.curr_weight = np.random.random_sample((phi_matrix.shape[1], n_class))
        old_weight = np.zeros((phi_matrix.shape[1], n_class))
        for i in range(max_iter):
            if np.allclose(old_weight, self.curr_weight): break
            old_weight = self.curr_weight
            grad_matrix, cost = self.gradient_mat(phi_matrix, old_weight, target_vec, reg_param_lambda, n_class)
            self.curr_weight -= l_rate * grad_matrix
            self.grad_matrix.append(grad_matrix)
            self.cost_iterations.append(i)
            self.cost_history.append(cost)

    def fit(self, X: np.ndarray, target_vec: np.ndarray, n_class: int) -> 'GradientDescentOptimizer':
        self.curr_weight = X
        self.grad_descent(
            X, self.l_rate, self.max_iteration, target_vec, self.reg_param_lambda, n_class)
        self.is_fitted_ = True
        return self
    

class LogisticRegressionFromScratch(BaseEstimator, TransformerMixin):

    def __init__(self, n_class) -> None:
        self.l_rate = CONFIG["l_rate"]
        self.max_iteration = CONFIG["max_iteration"]
        self.reg_param_lambda = CONFIG["reg_param_lambda"]
        self.weights = []
        self.n_class = n_class
        self.opt = GradientDescentOptimizer()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionFromScratch':
        target_vec = compute_target_vector(y, self.n_class)
        logger.info(f"y before : {y.shape} -  target vector shape: {target_vec.shape} - n_class {self.n_class}") 
        self.opt = self.opt.fit(X, target_vec, self.n_class)
        self.weights = self.opt.curr_weight
        logger.info(f'weights after fitting - {self.weights.shape}')
        self.is_fitted_ = True
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X, y)

    def transform(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionFromScratch':
        check_is_fitted(self)
        return self
        
    def summary(self) -> str:
        return "summary"

    def predict(self, X: np.ndarray, y: np.ndarray=None) -> List:
        logger.info(f"predict is being called on test_dataset")
        y_log_t = (X @ self.weights)
        logger.info(f"Predictions {y_log_t.shape}")
        y_logistic = custom_softmax(y_log_t)
        logger.info(f"custom softmaxed {y_logistic.shape}")
        mapped_y_pred = mapping_prob_to_class(y_logistic)
        return mapped_y_pred

    def plot_convergence(self) -> None:
        plt.plot(self.opt.cost_iterations, self.opt.cost_history)
        plt.title('Cost Function')
        plt.xlabel('Iterations')
        plt.ylabel('J_cost')
        plt.show(block=False) # non blocking show
        plt.pause(2) # pause for 5 seconds
        plt.close() # close the plot
        
    def save_checkpoint(self, filename: str) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        models_dir = os.path.join(parent_dir, "models")
        file_path = os.path.join(models_dir, filename)
        np.savez(file_path, weights=self.weights)
