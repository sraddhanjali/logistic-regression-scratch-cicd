import yaml
import pickle
import os
import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient
import socket
import dvc.api
import subprocess
import time
from metaflow import FlowSpec, step, batch
from ml_model import LogisticRegressionFromScratch
from sklearn.metrics import accuracy_score
from ml_datapipeline import DataPipeline
from pathlib import Path


# class ModelTrainerFlow(FlowSpec):

#     @step
#     def load_config(self):
#         self.config = self.load_config("config.yml")
#         self.max_iteration = self.config["max_iteration"]
#         self.mlflow_uri = self.config["mlflow"]["tracking_uri"]
#         self.mlflow_experimentname = self.config["mlflow"]["experiment_name"]
#         self.mlflow_port = self.config["mlflow"]["default_port"]
#         self.model_path = self.config["training"]["model_path"]
#         self.requested_dataset = self.config["data_pipelines"]["requested_dataset"]
#         self.dvc_remote = self.config["dvc"]["remote_storage"]

class ModelTrainer:
    def __init__(self, config_path="config.yml"):
        self.config = self.load_config(config_path)
        self.max_iteration = self.config["max_iteration"]
        self.mlflow_uri = self.config["mlflow"]["tracking_uri"]
        self.mlflow_experimentname = self.config["mlflow"]["experiment_name"]
        self.mlflow_port = self.config["mlflow"]["default_port"]
        self.model_path = self.config["training"]["model_path"]
        self.requested_dataset = self.config["data_pipelines"]["requested_dataset"]
        self.dvc_remote = self.config["dvc"]["remote_storage"]

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def request_data(self):
        """Request dataset dynamically from the pipeline."""
        pipeline = DataPipeline()
        return pipeline.get_data(dataset=self.requested_dataset)  # Returns (X_train, X_test, y_train, y_test)

    @staticmethod
    def find_free_port(default_port=5000):
        """Find an available port if the default is taken."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("localhost", default_port)) == 0:  # Port is taken
                s.close()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", 0))  # Bind to an available ephemeral port
                free_port = s.getsockname()[1]
                return s.getsockname()[1]
            else:
                free_port = default_port  # Default port is available
        return free_port  # Port 5000 is free
    
    def give_mlflow_time_to_start(self, port):
        for _ in range(10):  # Retry for ~10 seconds
            time.sleep(2)
            import requests
            try:
                if requests.get(f"http://localhost:{port}").status_code == 200:
                    print(f"✅ MLflow is running on port {port}")
                    return
            except requests.ConnectionError:
                print("❌ MLflow did not start in time. Check mlflow.log for errors.")
            # time.sleep(5)  # Give MLflow some time to start
        
        exit(1)
    
    def start_mlflow_server(self):
        """Start MLflow server in the background and log output."""
        log_file = open("mlruns/mlflow.log", "a")
        process = subprocess.Popen([
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", f"{self.mlflow_port}"
        ], stdout=log_file, stderr=log_file, start_new_session=True)
        print(f"🚀 Starting MLflow tracking server on port {self.mlflow_port}...")

        # self.give_mlflow_time_to_start(self.mlflow_port)
        print(f"🚀 Started MLflow tracking server on port {self.mlflow_port}...")
        return process


    def train(self, X_train, X_test, y_train, y_test):
        """Train and save model with MLflow & DVC."""
        
        self.start_mlflow_server()
        mlflow.set_tracking_uri(f"http://localhost:{self.mlflow_port}")
        exp = mlflow.set_experiment(self.mlflow_experimentname)       

        print(f'experiment starting {exp.experiment_id}')
        with mlflow.start_run():
            print(f"starting the run")
            model = LogisticRegressionFromScratch(n_class=np.sum(np.unique(y_train)))
            print(f"Fitting model")
            print(X_train, type(X_train), X_train.shape)
            model.fit_transform(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_artifact(os.path.join(os.getcwd(), "config.yml"))
            # Log params & metrics
            mlflow.log_param("dataset", self.requested_dataset)
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_metric("accuracy", acc)

            # Save model
            model_dir = f"models/{self.requested_dataset}/"
            os.makedirs(model_dir, exist_ok=True)
            mlflow.log_artifact(model_dir)

            model_file = os.path.join(os.getcwd(), self.config["model"]["model_dir"], self.requested_dataset + "_" + self.config["model"]["training_model"])
            
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_file)
            mlflow.log_artifact(self.dvc_remote)
            print(f"Model saved at {model_file} with accuracy: {acc}")
            # Track model & dataset with DVC    
            self.version_with_dvc(model_file)

    def version_with_dvc(self, model_file):
        """Use DVC to version the model and data."""
        os.system("dvc init")
        os.system(f"dvc remote add -d myremote {self.dvc_remote}")

        for f in os.listdir(os.path.join(os.getcwd(), self.config["data_dir"], self.requested_dataset)):
            os.system(f"dvc add {f}")
            os.system(f"git add {f}.dvc .gitignore")
        os.system("git commit -m 'Version dataset'")
        os.system("dvc push")

        # Track model
        os.system(f"dvc add {model_file}")
        os.system(f"git add {model_file}.dvc")
        os.system("git commit -m 'Version model'")
        os.system("dvc push")

    def run(self):
        """Execute model training and versioning."""
        X_train, X_test, y_train, y_test = self.request_data()
        print(f"Dataset retrieved")
        self.train(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
