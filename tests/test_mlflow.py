import mlflow
import subprocess
from mlflow import log_metric, log_param, log_artifact

# Add this at the start of your script
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
except Exception:
    exit


if __name__ == "__main__":
    log_param("threshold", 3)
    log_param("verbosity", "DEBUG")

    log_metric("timestamp", 1000)
    log_metric("TTC", 33)

    log_artifact("produced-dataset.csv")