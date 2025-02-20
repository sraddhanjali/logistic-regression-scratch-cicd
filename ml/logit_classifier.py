from typing import Tuple, List, Optional

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from config import tuning_params, non_tuning_params
import core


class PreprocessDataTransformer:
    # here init is for learning hyperparameters
    def __init__(self, mean_vars: Optional[List[Tuple[float, float]]] = None) -> None:
        self.mean_vars = mean_vars if mean_vars is not None else {}

    # Pipeline expects data through fit and transform
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PreprocessDataTransformer':
        return self

    def transform(self, X: np.ndarray) -> Tuple[List[np.ndarray], int]:
        descriptors, mean_vars = core.preprocess_data(X, self.mean_vars)
        self.mean_vars = mean_vars
        # print("Preprocess transform is called", X.shape, type(descriptors), len(descriptors), d_shape)
        return descriptors


class PhiMatrixTransformer:
    def __init__(self) -> None:
        self.phi_matrix = None
        pass

    def fit(self, X: Tuple[List[np.ndarray], int], y: Optional[np.ndarray] = None) -> 'PhiMatrixTransformer':
        return self

    def transform(self, X: np.ndarray) -> np.matrix:
        # print("the PhiMatrix Transform got called", X[1])
        descriptors, d_shape = X
        return core.create_phi_matrix(descriptors, d_shape, non_tuning_params["complexities"])


class GradientDescentOptimizer:
    def __init__(self) -> None:
        self.l_rate = tuning_params["l_rate"]
        self.max_iteration = tuning_params["max_iteration"]
        self.reg_param_lambda = tuning_params["reg_param_lambda"]
        self.cost_history = None
        self.cost_iterations = None
        self.grad_matrix_history = None
        self.phi_matrix = None
        self.curr_weight: Optional[np.ndarray] = None

    def fit(self, X: np.matrix, y: np.ndarray) -> 'GradientDescentOptimizer':
        # print("Gradient Descent fit got called", y.shape)
        _n_class = int(non_tuning_params.get('n_class', 3))
        target_vec = core.compute_target_vector(y, n_class=_n_class)
        self.curr_weight = np.random.random_sample((X.shape[1], _n_class))
        self.curr_weight, self.cost_history, self.cost_iterations, self.grad_matrix_history = core.grad_descent(
            X, self.curr_weight, self.l_rate, self.max_iteration, target_vec, self.reg_param_lambda, n_class=_n_class)
        # print("this fit is being called and weights are ", len(self.curr_weight))
        return self
    

class LogisticRegressionFromScratch(BaseEstimator):

    
    def __init__(self, l_rate, max_iteration, reg_param_lambda) -> None:
        # maintain state of learnable parameters over training etc.
        self.l_rate = l_rate
        self.max_iteration = max_iteration
        self.reg_param_lambda = reg_param_lambda
        self.pipeline = Pipeline([
        ('preprocess', PreprocessDataTransformer()),
        ('phi_matrix', PhiMatrixTransformer()),
        ('gradient_descent', GradientDescentOptimizer()),
    ])
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
        y_logistic = core.custom_softmax(y_log_t, along_axis=1)
        print(y_logistic.shape)
        mapped_y_pred = core.mapping_prob_to_class(y_logistic)
        return mapped_y_pred

    def plot_convergence(self) -> None:
        core.plot_cost_function_convergence(self.cost_history, self.cost_iterations)

    def save_checkpoint(self, filename: str) -> None:
        np.savez(filename, weights=self.pipeline.named_steps['gradient_descent'].curr_weight)
