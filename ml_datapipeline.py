import yaml
import pandas as pd
import numpy as np
import os

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

import sys
sys.path.append(".")  # Add folder to Python path
from utils import preprocessing as pp
from utils import features as feat
     
class DataPipeline:
    def __init__(self, scale: bool = True, augment_feature: bool = False, config_path: str="config.yml"):
        """Initialize pipeline with dataset triggers."""
        self.config = self.load_config(config_path)
        self.scaler = pp.ScalerTransform() if scale else None
        self.feat = None
        if self.config["m1"] and augment_feature:
            self.feat = feat.PhiMatrixTransformer(polynomial_degree=int(self.config["m1"]))
        self.seed = int(self.config["seed"])
        self.test_size = float(self.config["test_size"])
        return self

    def load_config(self, config_path: str):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def generate_synthetic_data(self):
        """Generate synthetic data for binary classification."""
        np.random.seed(self.seed)
        X = np.random.rand(1000, 5)
        y = np.random.randint(0, 2, 1000) # Binary labels (0/1)
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)]), pd.Series(y)

    @staticmethod
    def load_real_datasets(dataset_name):
        """Load predefined datasets."""
        if dataset_name == "iris":
            data = load_iris()
        elif dataset_name == "digits":
            data = load_digits()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        df = pd.DataFrame(data.data, columns=[f"feature_{i}" for i in range(data.data.shape[1])])
        df["target"] = data.target
        return df

    def detect_new_data(self):
        """Check for new datasets & process them."""
        datasets = ["synthetic", "iris", "digits"]
        for dataset in datasets:
            print(f"Processing dataset: {dataset}")
            if dataset == "synthetic":
                X, y = self.generate_synthetic_data()
                df = pd.concat([X, y.rename("target")], axis=1)
            else:
                df = DataPipeline.load_real_datasets(dataset)

            X_train, X_test, y_train, y_test = self.preprocess_and_split(df)
            self.store_data(X_train, X_test, y_train, y_test, dataset)

    def preprocess_and_split(self, df):
        """Preprocess features and split dataset."""
        X = df.drop(columns=["target"])
        y = df["target"]
        if self.scaler:
            X = self.scaler.fit_transform(X)
        if self.feat:
            X = self.feat.fit_transform(X)
        return train_test_split(X, y, test_size=self.test_size, random_state=self.seed)

    def store_data(self, X_train, X_test, y_train, y_test, dataset_name):
        """Store preprocessed dataset."""
        output_path = f"data/{dataset_name}"
        os.makedirs(output_path, exist_ok=True)
        pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1).to_csv(f"{output_path}/train.csv", index=False)
        pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1).to_csv(f"{output_path}/test.csv", index=False)

    def run(self):
        """Run the pipeline for all dataset triggers."""
        self.detect_new_data()
        print("Data Pipeline Execution Completed.")

    def get_data(self):
        csv_f = os.path.join(self.config["output_path"], self.config["source"], "train.csv")
        return self.preprocess_and_split(pd.read(csv_f))

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
