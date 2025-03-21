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
import glob

import os
from functools import wraps
import subprocess

def ensure_dvc_initialized(func):
    """Decorator to initialize DVC repo if not already."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not os.path.exists(".dvc"):
            print("🛠 Initializing DVC...")
            subprocess.run(["dvc", "init"], check=True)
        else:
            print("✅ DVC already initialized.")
        return func(*args, **kwargs)
    return wrapper

def skip_if_tracked(func):
    """Skip DVC add if file is already tracked."""
    @wraps(func)
    def wrapper(filepath, *args, **kwargs):
        if os.path.exists(filepath + ".dvc"):
            print(f"✅ File '{filepath}' already tracked by DVC. Skipping.")
            return
        return func(filepath, *args, **kwargs)
    return wrapper

def ensure_remote_not_exists(func):
    """Decorator factory to skip remote creation if it already exists."""
    @wraps(func)
    def wrapper(remote_name, *args, **kwargs):
        result = subprocess.run(["dvc", "remote", "list"], capture_output=True, text=True)
        if remote_name in result.stdout:
            print(f"✅ Remote '{remote_name}' already exists. Skipping.")
            return
        return func(*args, **kwargs)
    return wrapper

class DVCManager:
    def __init__(self, dvc_config):
        self.dvc_config = dvc_config

    @skip_if_tracked
    def version_data(self, f_path):
        subprocess.run(["dvc", "add", f_path], check=True)
        subprocess.run(["git", "add", f"{f_path}.dvc", ".gitignore"])
        os.system("git commit -m 'Version dataset'")
        os.system("dvc push")

    @ensure_dvc_initialized
    @ensure_remote_not_exists("myremote")
    def add_remote(self):
        name = "myremote"
        url = self.dvc_config["remote_storage"]
        subprocess.run(["dvc", "remote", "add", "-d", name, url], check=True)
        print(f"🚀 DVC remote '{name}' added.")

    @ensure_dvc_initialized
    @ensure_remote_not_exists("localremote")
    def add_local_remote(self):
        name = "localremote"
        url = self.dvc_config["local_storage"]
        subprocess.run(["dvc", "remote", "add", "-d", name, url], check=True)
        print(f"🚀 DVC local remote '{name}' added.")
    
    def setup_remote(self):
        if bool(self.dvc_config["remote"]):
            self.add_local_remote()
        else:
            self.add_remote()
            
    def version(self, d):
        subprocess.run(["dvc", "add", f"{d}"])
        subprocess.run(["git", "add", f"{d}"])
    
    def push_all(self):
        subprocess.run(["dvc", "push"])

class ModelTrainer:
    def __init__(self, config_path="config.yml"):
        self.config = self.load_config(config_path)
        self.max_iteration = self.config["max_iteration"]
        self.mlflow_uri = self.config["mlflow"]["tracking_uri"]
        self.mlflow_experimentname = self.config["mlflow"]["experiment_name"]
        self.mlflow_port = self.config["mlflow"]["default_port"]
        self.model_path = self.config["training"]["model_path"]
        self.requested_dataset = self.config["data_pipelines"]["requested_dataset"]

    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def request_data(self):
        """Request dataset dynamically from the pipeline."""
        pipeline = DataPipeline()
        return pipeline.get_data(dataset=self.requested_dataset)  # Returns (X_train, X_test, y_train, y_test)
    
    def start_mlflow_server(self):
        """Start MLflow server in the background and log output."""
        log_file = open("mlruns/mlflow.log", "a")
        process = subprocess.Popen([
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", f"{self.mlflow_port}"
        ], stdout=log_file, stderr=log_file, start_new_session=True)
        print(f"🚀 Starting MLflow tracking server on port {self.mlflow_port}...")
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

            #dataset path
            dataset_path = os.path.join(self.config["data_dir"], self.requested_dataset)
            mlflow.log_param("dataset_path", dataset_path)

            # Save model
            model_dir = f"models/{self.requested_dataset}/"
            os.makedirs(model_dir, exist_ok=True)
            mlflow.log_artifact(model_dir)
    
            model_file = os.path.join(os.getcwd(), self.config["model"]["model_dir"], self.requested_dataset + "_" + self.config["model"]["training_model"])
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_file)
            print(f"Model saved at {model_file} with accuracy: {acc}")

            self.version_with_dvc(model_file, dataset_path)


    def version_with_dvc(self, model, data):
        """Use DVC to version the model and data."""
        dvc_manager = DVCManager(config=self.config)
        dvc_manager.setup_remote()
        dvc_manager.version(data)
        dvc_manager.version(model)
        dvc_manager.push_all()

    def run(self):
        """Execute model training and versioning."""
        X_train, X_test, y_train, y_test = self.request_data()
        print(f"Dataset retrieved")
        self.train(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
