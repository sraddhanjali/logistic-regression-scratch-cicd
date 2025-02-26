import numpy as np

import mlflow

# Add this at the start of your script
mlflow.set_tracking_uri("http://127.0.0.1:5000")
from mlflow import log_metric, log_param, log_artifact


seed = 100
random_state = np.random.RandomState(seed=100)
# Not tunable parameters
n_samples = 1000
n_class = 3
fold = 2

m1 = 1
rate1 = 0.05
max_iters = 500
lamda1 = 0.00001
polys = [i for i in range(1, m1 + 1)]
test_size = 0.3

tuning_params = {
    'l_rate': rate1,
    'max_iteration': max_iters,
    'reg_param_lambda': lamda1,
}

non_tuning_params = {
    'complexities': polys, 
    'n_samples': n_samples,
    'n_class': n_class,
    'n_folds': fold,
    'n_random_state': random_state,
    'test_size': test_size
}


log_param("l_rate", rate1)
log_param("max_iteration", max_iters)
log_param("reg_param_lambda", lamda1)
log_param("polynomial_degree", m1)
log_param("complexitiies", polys)
log_param("n_samples", n_samples)
log_param("n_folds", fold)
log_param("n_random_state", random_state)
log_param("test_size", test_size)


from time import time

log_metric("timestamp", time())
