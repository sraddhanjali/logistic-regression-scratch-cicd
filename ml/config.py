# Not tunable parameters
n_samples = 1000
n_class = 3
fold = 2
random_state = 42


m1 = 1
rate1 = 0.05
max_iters = 500
lamda1 = 0.00001
polys = [i for i in range(1, m1 + 1)]


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
    'test_size': 0.30   
}
