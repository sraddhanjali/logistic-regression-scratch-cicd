import time
import pdb
import pandas as pd
import numpy as np
import sklearn

import matplotlib.colors as mc
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Optional, Any
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

from config import tuning_params, non_tuning_params


sklearn.set_config(enable_metadata_routing=True)


def custom_softmax(unsoftmaxed_data, along_axis=0):
    """ Computes the softmax function to data along a particular axis (only for rows, columns).
    :param unsoftmaxed_data: data to be applied softmax to. (ndarray)
    :param along_axis: axis along which to apply softmax;
           along_axis = 0 applies the function column wise (default value).
           along_axis = 1 applies the function row wise.
    :return: softmaxed data (ndarray).
    """
    def softmax_func(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    return np.apply_along_axis(softmax_func, along_axis, unsoftmaxed_data)


def create_powers_desc(descriptors, complexities):
    """ Adds powers of descriptors using values in polys.
    :param descriptors: a list of descriptors.
    :param complexities: a list of complexities.
    :return: iterable for powers.
    """
    for degree in complexities[1:]:
        for d in descriptors:
            r = np.power(d, degree)
            yield r.T


def create_combination_desc(descriptors, complexities):
    """ Build combinations of descriptors.
    :param descriptors: a list of descriptors.
    :param complexities: a list of polynomial complexities.
    :param d_shape: number of instances in data.
    :return: iterable for combination of descriptors.
    """
    d_shape = descriptors[0].shape[0]
    for degree in complexities:
        indices = [i for i in range(len(descriptors))]
        # print(indices)
        iterab = combinations(indices, degree)
        for it in iterab:
            # print(it, len(it), type(it))
            mult = np.ones(d_shape)
            for i in it:
                # print(mult.shape, d_shape, descriptors[i].shape)
                mult *= descriptors[i][:, 0]
            yield mult.T


def create_phi_matrix(descriptors, complexities):
    """ Build a phi data from descriptors.
    :param descriptors: a list of descriptors.
    :param d_shape: number of instances in data.
    :param complexities: a list of polynomial complexities.
    :return: matrix of features built from desc as a list.
    """
    # print("create_phi_matrix is called", type(descriptors), len(descriptors), d_shape)
    # print("complexities", complexities)
    phi = []
    one = np.ones(descriptors[0].shape[0])
    phi.append(one)
    for val in create_combination_desc(descriptors, complexities):
        phi.append(val)
    for val in create_powers_desc(descriptors, complexities):
        phi.append(val)
    return np.matrix(np.vstack(phi)).T


def compute_target_vector(y_original, _n_class):
    """ Convert a 1D class labels to numerical encodings.
    :param y_original: 1D class labels.
    :param n_class: number of classes.
    :return: numerical encodings (ndarray).
    """

    assert _n_class == len(np.unique(y_original)), "Number of classes should be equal to the unique classes in the data"
    if _n_class == 2:
        y_encoded = np.zeros((y_original.shape[0], 2))
        for i in range(y_original.shape[0]):
            if y_original[i] == 1:
                y_encoded[i, 1] = 1
            elif y_original[i] == 0:
                y_encoded[i, 0] = 1
    else:
        if y_original.ndim == 1 or y_original.shape[1] == 1:
            labelencoder_y = LabelBinarizer()
            y_encoded = labelencoder_y.fit_transform(y_original)
        else:
            y_encoded = y_original
    return y_encoded
    
    
def gradient_mat(phi_matrix, old_weight, target_vec, reg_param_lambda, _n_class):
    """ Computes the cost function and the gradient matrix.
    :param phi_matrix: matrix containing features built from original descriptors from data.
    :param old_weight: old weight matrix.
    :param n_class: number of classes.
    :param target_vec: target encoded values.
    :param reg_param_lambda: regularization parameter.
    :return: error gradient matrix and cost Function (float).
    """
    y = phi_matrix * old_weight
    sample_size = y.shape[0]
    y = custom_softmax(y, along_axis=1)
    b = (1/sample_size) * reg_param_lambda * np.sum(np.square(old_weight))
    tar_vector_sh = (target_vec.reshape((sample_size * _n_class), 1)).T
    y_sh = y.reshape((sample_size * _n_class), 1)
    cost_val = -(tar_vector_sh * np.log(y_sh)) / sample_size + b
    diff = y - target_vec
    grad_matrix = ((phi_matrix.T * diff) + reg_param_lambda * old_weight) / sample_size
    return grad_matrix, cost_val


def grad_descent(phi_matrix, curr_weight, l_rate, max_iteration, target_vec, reg_param_lambda, _n_class):
    """Optimize weights using gradient descent algorithm with stopping criterions.
    :param phi_matrix: matrix containing features built from original descriptors from data.
    :param curr_weight: current weight matrix.
    :param l_rate: learning rate.
    :param max_iteration: maximum iterations to go for while updating weights.
    :param target_vec: target encoded values.
    :param reg_param_lambda: regularization parameter.
    :param n_class: number of classes - default arg=3.
    :return: new weights, cost list, total iterations, gradient matrix history.
    """
    old_weight = np.zeros((phi_matrix.shape[1], _n_class))
    iters = 0
    cost_history = []
    cost_iterations = []
    grad_matrix_history = []
    while not np.allclose(old_weight, curr_weight) and iters < max_iteration:
        old_weight = curr_weight
        grad_matrix, cost_val = gradient_mat(phi_matrix, old_weight, target_vec, reg_param_lambda, _n_class)
        curr_weight = curr_weight - l_rate * grad_matrix
        cost_history.append(cost_val)
        grad_matrix_history.append(grad_matrix)
        cost_iterations.append(iters)
        iters = iters + 1
    return curr_weight, cost_history, cost_iterations, grad_matrix_history


def standardize_data(org_data):
    """ Use mean and variance of data to standardize the data.
    :param org_data: data (1D) array.
    :return: standardized array, with its mean and variance.
    """
    mean_ = np.mean(org_data)
    var_ = np.std(org_data)
    stand_data = (org_data - mean_) / var_
    return stand_data, mean_, var_


def mapping_prob_to_class(y_logistic):
    """
    :param y_logistic: encoded labels.
    :return: list of class labels obtained from encoded labels.
    """
    return [np.argmax(row) for row in y_logistic]


def accuracy(y_pred, y_original):
    """ Compute accuracy comparing matches between predicted and original class labels.
    :param y_pred: predicted class labels (list).
    :param y_original: original class labels (list).
    :return: accuracy percentage (float).
    """
    y_pred = np.asarray(y_pred)
    y_original = np.asarray(y_original)
    count = np.sum(y_pred == y_original)
    acc = (count/len(y_pred)) * 100
    return acc

def preprocess_data(descriptors, mean_vars):
### TODO come and fix this function
    if not mean_vars:
        mean = np.mean(descriptors, axis=1)
        var = np.std(descriptors, axis=1)
        descriptors = (descriptors - mean) / var
        mean_vars = [mean, var]
    else:
        mean, var = mean_vars
        print(mean)
        descriptors = (descriptors - mean) / var
    return descriptors, mean_vars


def plot_cost_function_convergence(cost_history, cost_iteration):
    for i in range(len(cost_iteration)):
        cost_history[i] = cost_history[i].tolist()
    cost_history = [i[0] for i in cost_history]
    plt.plot(cost_iteration, cost_history)
    plt.title('Cost Function')
    plt.xlabel('Iterations')
    plt.ylabel('J_cost')
    plt.show()


def average_accuracies(accuracy_dict):
    accuracies = list(accuracy_dict.values())
    n_accuracies = len(accuracies)
    return np.sum(accuracies)/n_accuracies


def averaging_times(time_dict):
    time_values = list(time_dict.values())
    n_time_values = len(time_values)
    return np.sum(time_values)/n_time_values


class PreprocessDataTransformer:
    def __init__(self, mean_vars: Optional[List[Tuple[float, float]]] = None) -> None:
        self.mean_vars = mean_vars if mean_vars is not None else {}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PreprocessDataTransformer':
        return self

    def transform(self, X: np.ndarray) -> Tuple[List[np.ndarray], int]:
        descriptors, mean_vars = preprocess_data(X, self.mean_vars)
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
        return create_phi_matrix(descriptors, d_shape, non_tuning_params["complexities"])


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
        target_vec = compute_target_vector(y, n_class=_n_class)
        self.curr_weight = np.random.random_sample((X.shape[1], n_class))
        self.curr_weight, self.cost_history, self.cost_iterations, self.grad_matrix_history = grad_descent(
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
        y_logistic = custom_softmax(y_log_t, along_axis=1)
        print(y_logistic.shape)
        mapped_y_pred = mapping_prob_to_class(y_logistic)
        return mapped_y_pred

    def plot_convergence(self) -> None:
        plot_cost_function_convergence(self.cost_history, self.cost_iterations)

    def save_checkpoint(self, filename: str) -> None:
        np.savez(filename, weights=self.pipeline.named_steps['gradient_descent'].curr_weight)

def hyperparameter_tuning(X: np.ndarray, y: np.ndarray) -> GridSearchCV:
    model = LogisticRegressionFromScratch(tuning_params["l_rate"], tuning_params["max_iteration"], tuning_params["reg_param_lambda"])
    model = model.fit(X, y)
    # clf = GridSearchCV(estimator=model, param_grid=tuning_params, cv=3, scoring='accuracy', verbose=True, n_jobs=-1)
    # clf_fit = clf.fit(X, y)
    # print("best_estimator ---- ", clf_fit.best_estimator_)
    print("score----", accuracy(model.predict(X), y))
    # print("best_score---", clf_fit.best_score_)
    return model


def make_synthetic_data(n_samples, n_classes, random_state, n_features=2):
    """ Make synthetic data to work for validation of implementation.
    :param n_samples: number of samples in data.
    :param n_classes: number of classes.
    :param n_features: number of features built from descriptors.
    :return: data (ndarray).
    """
    return make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=1.6, random_state=random_state)


def run_model(X, y):
    data = [X, y]
    for train_ind, test_ind in kf.split(data[0]):
        _fold += 1
        print("--------Running K fold for fold={0}.......".format(_fold))
        # print(train_ind, test_ind)
        X_train, X_test = data[0][train_ind], data[0][test_ind]
        y_train, y_test = data[1][train_ind], data[1][test_ind]
    
        model = hyperparameter_tuning(X_train, y_train)
        # print("shapes of train descriptors & their labels ----->")
        # print(X_train.shape, y_train.shape)
        # print("shapes of test descriptors & their labels ----->")
        # print(X_test.shape, y_test.shape)
        # print("shapes of y_train, y_test----->")
        # print(y_train.shape, y_test.shape)
        y_pred = model.predict(X_test)
        acc = accuracy(y_pred, y_test)
        # print("shapes of y_pred----->")
        # print(y_pred, len(y_pred))
        print(f"Accuracy for fold {_fold}: {acc}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    print(f"Best accuracy from KFold: {best_accuracy}")
    # best_model.plot_convergence()
    # best_model.save_checkpoint('best_model_kfold_checkpoint.npz')


def run_test_old_way(data=None):
    _l_rate = tuning_params.get('l_rate', 0.05)
    _max_iteration = tuning_params.get('max_iteration', 500)
    _reg_param_lambda = tuning_params.get('reg_param_lambda', 0.00001)

    # integer params - non tunable
    _n_class = int(non_tuning_params.get('n_class', 3))
    _n_samples = int(non_tuning_params.get('n_samples', 1000))
    _n_folds = int(non_tuning_params.get('n_folds', 2))
    _n_random_state = int(non_tuning_params.get('random_state', 42))
    _complexitites = non_tuning_params.get('complexities', [1])

    X, y = None, None

    if not data:
        data = make_synthetic_data(_n_samples, _n_class, random_state=_n_random_state)
        X, y = data[0], data[1]
    
    X, y = data[0], data[1]

    print(X.shape, y.shape)

    kf = KFold(n_splits=_n_folds, shuffle=True, random_state=_n_random_state)
    
    _fold = 0
    X_train = []
    X_test = []

    for train_index, test_index in kf.split(X[:, 0]):
        print("len of fold of train test indices", len(train_index), len(test_index))
        _fold += 1
        print("--------Running K fold for fold={0}.......".format(_fold))
        X_train.append(X[train_index])
        X_test.append(X[test_index])
        y_train, y_test = y[train_index], y[test_index]
        assert y_train.all() != None, "Target train vector should have more than one element"
        assert y_test.all() != None, "Target test vector should have more than one element"
        desc, mean_var_list = preprocess_data(X_train, {})
   
        desc_test, mean_var_list = preprocess_data(X_test, mean_var_list)
        phi_fin_train = create_phi_matrix(desc, _complexitites)
        phi_fin_test = create_phi_matrix(desc_test, _complexitites)

        tar_vector = compute_target_vector(y_train, _n_class)
        assert tar_vector.all() != None, "Target vector should have more than one element"
        cur_w = np.random.random_sample((phi_fin_train.shape[1], _n_class))


        # start time for training
        t0 = time.time()
        fin_w, cost_history, cost_iterations, g = grad_descent(phi_fin_train, cur_w, _l_rate, _max_iteration, tar_vector, _reg_param_lambda, _n_class)
        t1 = time.time()
        training_time = t1 - t0
        # end time for training

        # plot_cost_function_convergence(cost_history, cost_iterations)
        # start of testing
        t00 = time.time()
        y_logistic_train = (fin_w.T * phi_fin_train.T).T
        print(fin_w.shape, phi_fin_train.shape, y_logistic_train.shape)
        print("_______________________________-------")
        t11 = time.time()
        testing_time = t11 - t00
        # end of testing
        y_logistic_train = custom_softmax(y_logistic_train, along_axis=1)
        y_map_train = mapping_prob_to_class(y_logistic_train)

        # start time testing
        y_logistic_test = (fin_w.T * phi_fin_test.T).T
        # end time testing
        y_logistic_test = custom_softmax(y_logistic_test, along_axis=1)
        y_map_test = mapping_prob_to_class(y_logistic_test)

        # plot_confusion_matrix(y_test, y_map_test)
        acc_train = accuracy(y_map_train, y_train)
        print("Training accuracy = {0}".format(acc_train))
        acc_test = accuracy(y_map_test, y_test)
        print("Testing accuracy = {0}".format(acc_test))
        print("Training time = {0}".format(training_time))
        print("Testing time = {0}".format(testing_time))


def run_kfold(X, y):
    _l_rate = tuning_params.get('l_rate', 0.05)
    _max_iteration = tuning_params.get('max_iteration', 500)
    _reg_param_lambda = tuning_params.get('reg_param_lambda', 0.00001)

    # integer params - non tunable
    _n_class = int(non_tuning_params.get('n_class', 3))
    _n_samples = int(non_tuning_params.get('n_samples', 1000))
    _n_folds = int(non_tuning_params.get('n_folds', 2))
    _n_random_state = int(non_tuning_params.get('random_state', 42))

    data = make_synthetic_data(_n_samples, _n_class, random_state=_n_random_state)

    kf = KFold(n_splits=_n_folds, shuffle=True, random_state=_n_random_state)
    kf.get_n_splits(data[0])
    
    _fold = 0
    best_accuracy = 0
    best_model = None
    X_train = []
    X_test = []

    for train_index, test_index in kf.split(data[0][:, 0]):
        _fold += 1
        print("--------Running K fold for fold={0}.......".format(_fold))
        for i in range(data[0].shape[1]):
            X_train.append(data[0][:, i][train_index])
            X_test.append(data[0][:, i][test_index])

        mean_var_list = []
        y_train, y_test = data[1][train_index], data[1][test_index]
        assert y_train.all() != None, "Target train vector should have more than one element"
        assert y_test.all() != None, "Target test vector should have more than one element"
        desc, mean_var_list = preprocess_data(X_train, {})
        desc_test, mean_var_list = preprocess_data(X_test, mean_var_list)
        phi_fin_train = create_phi_matrix(desc, non_tuning_params["complexities"])
        phi_fin_test = create_phi_matrix(desc_test, non_tuning_params["complexities"])

        tar_vector = compute_target_vector(y_train, _n_class)
        assert tar_vector.all() != None, "Target vector should have more than one element"
        cur_w = np.random.random_sample((phi_fin_train.shape[1], _n_class))


        # start time for training
        t0 = time.time()
        fin_w, cost_history, cost_iterations, g = grad_descent(phi_fin_train, cur_w, _l_rate, _max_iteration, tar_vector, _reg_param_lambda, _n_class)
        t1 = time.time()
        training_time = t1 - t0
        # end time for training

        # plot_cost_function_convergence(cost_history, cost_iterations)
        # start of testing
        t00 = time.time()
        y_logistic_train = (fin_w.T * phi_fin_train.T).T
        print(fin_w.shape, phi_fin_train.shape, y_logistic_train.shape)
        print("_______________________________-------")
        t11 = time.time()
        testing_time = t11 - t00
        # end of testing
        y_logistic_train = custom_softmax(y_logistic_train, along_axis=1)
        y_map_train = mapping_prob_to_class(y_logistic_train)

        # start time testing
        y_logistic_test = (fin_w.T * phi_fin_test.T).T
        # end time testing
        y_logistic_test = custom_softmax(y_logistic_test, along_axis=1)
        y_map_test = mapping_prob_to_class(y_logistic_test)

        # plot_confusion_matrix(y_test, y_map_test)
        acc_train = accuracy(y_map_train, y_train)
        print("Training accuracy = {0}".format(acc_train))
        acc_test = accuracy(y_map_test, y_test)
        print("Testing accuracy = {0}".format(acc_test))
        print("Training time = {0}".format(training_time))
        print("Testing time = {0}".format(testing_time))


if __name__ == "__main__":
    
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
    
    print("--The code runs for synthetic dataset first and then for iris dataset next--")
    print("-----------For m={0} ------".format(m1))
    print("-----------For rate={0}-----".format(rate1))
    print("-----------For max_iters={0}----".format(max_iters))
    print("-----------For lambda={0}----".format(lamda1))
    
    print("-----------FOR SYNTHETIC DATASET---------")

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
        'n_random_state': random_state   
    }


    # run_kfold()
    