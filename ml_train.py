import yaml
import pickle
import os
import pandas as pd
import numpy as np
import mlflow
import dvc.api
from metaflow import FlowSpec, step, batch
from ml_model import LogisticRegressionFromScratch
from sklearn.metrics import accuracy_score
from ml_datapipeline import DataPipeline
from pathlib import Path
import sys
sys.path.append(".")
from utils.mlflow_manager import MLflowManager
from utils.dvc_manager import DVCManager
from utils.config import load_config, flatten_numeric_values
from functools import wraps


class ModelTrainer:
    def __init__(self, config_path="config.yml"):
        self.config_path = config_path
        self.config = load_config(self.config_path)
        self.max_iteration = self.config["max_iteration"]
        self.mlflow = MLflowManager(mlflow_config=self.config["mlflow"])
        self.model_path = self.config["training"]["model_path"]
        self.requested_dataset = self.config["data_pipelines"]["requested_dataset"]

    def request_data(self):
        """Request dataset dynamically from the pipeline."""
        pipeline = DataPipeline()
        return pipeline.get_data(dataset=self.requested_dataset)  # Returns (X_train, X_test, y_train, y_test)

    def train(self, X_train, X_test, y_train, y_test):
        """Train and save model with MLflow & DVC."""
        self.mlflow.start()
        with mlflow.start_run():
            print(f"starting the run")
            model = LogisticRegressionFromScratch(n_class=np.sum(np.unique(y_train)))
            print(f"Fitting model")
            print(X_train, type(X_train), X_train.shape)
            model.fit_transform(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            #dataset path
            dataset_path = os.path.join(self.config["data_dir"], self.requested_dataset)
            # Save model
            model_dir = os.path.join(os.getcwd(), self.config["model"]["model_dir"], self.requested_dataset)
            os.makedirs(model_dir, exist_ok=True)
    
            model_file = os.path.join(model_dir, self.requested_dataset + "_" + self.config["model"]["training_model"])
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved at {model_file} with accuracy: {acc}")

            info = {
                "params": {
                    "model_dir": model_dir,
                    "requested_data": self.requested_dataset,
                    "classifier": "LogisticRegression"
                },
                "metrics": {
                    "accuracy": acc,
                }, 
                "artifacts": [
                    model_dir,
                    model_file, 
                    dataset_path
                ],
                "configs": flatten_numeric_values(self.config)
            }
            self.mlflow.log(info)
            self.version_with_dvc(model_file, dataset_path)
            self.mlflow.stop()

    def version_with_dvc(self, model, data):
        """Use DVC to version the model and data."""
        DVCManager().setup_remote(dvc_config=self.config["dvc"])
        DVCManager().version(d=model)
        DVCManager().version(d=data)
        DVCManager().commit_and_push()

    def run(self):
        """Execute model training and versioning."""
        X_train, X_test, y_train, y_test = self.request_data()
        print(f"Dataset retrieved")
        self.train(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
