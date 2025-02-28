import mlflow
import mlflow.sklearn
from ml_model import grad_descent
from utils.utils import compute_target_vector

def train_with_mlflow(X_train, y_train, complexities, l_rate, max_iter, reg_param_lambda, n_class):
    with mlflow.start_run():
        mlflow.log_params({"learning_rate": l_rate, "max_iterations": max_iter, "lambda": reg_param_lambda})

        target_vec = compute_target_vector(y_train, n_class)
        cur_w = np.random.randn(X_train.shape[1], n_class)
        final_weights, cost_history = grad_descent(X_train, n_class, cur_w, l_rate, max_iter, target_vec, reg_param_lambda)
        
        mlflow.log_metric("final_cost", cost_history[-1])
        mlflow.sklearn.log_model(final_weights, "model")
