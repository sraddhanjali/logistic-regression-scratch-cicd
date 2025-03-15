import pytest
import numpy as np
from utils.utils import compute_target_vector
from ml_model import GradientDescentOptimizer, LogisticRegressionFromScratch


def test_gradient_optimizer_initialization():
    optimizer = GradientDescentOptimizer()
    assert optimizer.l_rate > 0
    assert optimizer.max_iteration > 0
    assert optimizer.reg_param_lambda >= 0
    assert optimizer.curr_weight is None


def test_gradient_optimizer_fit():
    optimizer = GradientDescentOptimizer()
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, 100)  # 3-class classification
    n_class = 3
    target_vec = compute_target_vector(y, n_class)
    optimizer.fit(X, target_vec, n_class)

    assert optimizer.is_fitted_
    assert optimizer.curr_weight is not None
    assert len(optimizer.cost_history) > 0


def test_gradient_optimizer_gradient_mat():
    optimizer = GradientDescentOptimizer()
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, 100)  # 3-class classification
    n_class = 3
    target_vec = compute_target_vector(y, n_class)
    old_weight = np.random.rand(X.shape[1], n_class)
    grad_matrix, cost = optimizer.gradient_mat(
        X, old_weight, target_vec, optimizer.reg_param_lambda, n_class
    )

    assert grad_matrix.shape == old_weight.shape
    assert isinstance(cost, float)


def test_logistic_regression_initialization():
    lr = LogisticRegressionFromScratch(n_class=3)
    assert lr.n_class == 3
    assert lr.opt is not None


def test_logistic_regression_fit():
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, 100)  # 3-class classification
    lr = LogisticRegressionFromScratch(n_class=3)
    lr.fit(X, y)

    assert lr.is_fitted_
    assert lr.weights is not None


def test_logistic_regression_predict():
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, 100)  # 3-class classification
    lr = LogisticRegressionFromScratch(n_class=3)
    lr.fit(X, y)
    predictions = lr.predict(X)

    assert len(predictions) == y.shape[0]
    assert isinstance(predictions, list)


def test_logistic_regression_plot_convergence():
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, 100)  # 3-class classification
    try:
        lr = LogisticRegressionFromScratch(n_class=3)
        lr.fit(X, y)
        lr.plot_convergence()
    except (ValueError, TypeError, RuntimeError) as e:
        pytest.fail(f"plot_convergence raised an unexpected error: {e}")


def test_logistic_regression_save_checkpoint(tmp_path):
    lr = LogisticRegressionFromScratch(n_class=3)
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, 100)  # 3-class classification
    lr.fit(X, y)

    filename = tmp_path / "test_checkpoint.npz"
    lr.save_checkpoint(str(filename))

    assert filename.exists()


def test_invalid_shapes_in_gradient_mat():
    optimizer = GradientDescentOptimizer()
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 3, 100)  # 3-class classification
    n_class = 3
    target_vec = compute_target_vector(y, n_class)
    old_weight = np.random.rand(X.shape[1] + 1, n_class)  # Wrong shape

    with pytest.raises(ValueError):
        optimizer.gradient_mat(
            X, old_weight, target_vec, optimizer.reg_param_lambda, n_class
        )
