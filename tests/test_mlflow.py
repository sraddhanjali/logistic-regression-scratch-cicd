import mlflow

# Add this at the start of your script
mlflow.set_tracking_uri("http://127.0.0.1:5000")

from mlflow import log_metric, log_param, log_artifact

if __name__ == "__main__":
    log_param("threshold", 3)
    log_param("verbosity", "DEBUG")

    log_metric("timestamp", 1000)
    log_metric("TTC", 33)

    log_artifact("produced-dataset.csv")