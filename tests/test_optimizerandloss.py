import pytest
import numpy as np
from utils.utils import compute_target_vector, custom_softmax
from ml_model import GradientDescentOptimizer, LogisticRegressionFromScratch 

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, 100)  # 3-class classification
    return X, y

@pytest.fixture
def optimizer():
    return GradientDescentOptimizer()

@pytest.fixture
def logistic_regression():
    return LogisticRegressionFromScratch(n_class=3)

def test_gradient_optimizer_initialization(optimizer):
    assert optimizer.l_rate > 0
    assert optimizer.max_iteration > 0
    assert optimizer.reg_param_lambda >= 0
    assert optimizer.curr_weight is None

def test_gradient_optimizer_fit(optimizer, sample_data):
    X, y = sample_data
    n_class = 3
    target_vec = compute_target_vector(y, n_class)
    optimizer.fit(X, target_vec, n_class)
    
    assert optimizer.is_fitted_
    assert optimizer.curr_weight is not None
    assert len(optimizer.cost_history) > 0

def test_gradient_optimizer_gradient_mat(optimizer, sample_data):
    X, y = sample_data
    n_class = 3
    target_vec = compute_target_vector(y, n_class)
    old_weight = np.random.rand(X.shape[1], n_class)
    grad_matrix, cost = optimizer.gradient_mat(X, old_weight, target_vec, optimizer.reg_param_lambda, n_class)

    assert grad_matrix.shape == old_weight.shape
    assert isinstance(cost, float)

def test_logistic_regression_initialization(logistic_regression):
    assert logistic_regression.n_class == 3
    assert logistic_regression.opt is not None

def test_logistic_regression_fit(logistic_regression, sample_data):
    X, y = sample_data
    logistic_regression.fit(X, y)

    assert logistic_regression.is_fitted_
    assert logistic_regression.weights is not None

def test_logistic_regression_predict(logistic_regression, sample_data):
    X, y = sample_data
    logistic_regression.fit(X, y)
    predictions = logistic_regression.predict(X)

    assert len(predictions) == y.shape[0]
    assert isinstance(predictions, list)

def test_logistic_regression_plot_convergence(logistic_regression, sample_data):
    X, y = sample_data
    logistic_regression.fit(X, y)

    try:
        logistic_regression.plot_convergence()
    except Exception as e:
        pytest.fail(f"plot_convergence raised an exception: {e}")

def test_logistic_regression_save_checkpoint(logistic_regression, sample_data, tmp_path):
    X, y = sample_data
    logistic_regression.fit(X, y)
    
    filename = tmp_path / "test_checkpoint.npz"
    logistic_regression.save_checkpoint(str(filename))

    assert filename.exists()

def test_invalid_shapes_in_gradient_mat(optimizer, sample_data):
    X, y = sample_data
    n_class = 3
    target_vec = compute_target_vector(y, n_class)
    old_weight = np.random.rand(X.shape[1] + 1, n_class)  # Wrong shape

    with pytest.raises(ValueError):
        optimizer.gradient_mat(X, old_weight, target_vec, optimizer.reg_param_lambda, n_class)
