import yaml
import pandas as pd
import numpy as np
import os
import glob

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split

import sys
sys.path.append(".")  # Add folder to Python path
from utils import preprocessing as pp
from utils import features as feat
from typing import Any, Tuple

     
class DataPipeline:
    def __init__(self, scale: bool = True, augment_feature: bool = False, config_path: str="config.yml"):
        """Initialize pipeline with dataset triggers."""
        self.config = self.load_config(config_path)
        self.scaler = pp.ScalerTransform() if scale else None
        self.synthetic_data_info = (self.config["n_class"], self.config["n_feature"], self.config["n_sample"])
        self.feat = None
        if self.config["m1"] and augment_feature:
            self.feat = feat.PhiMatrixTransformer(polynomial_degree=int(self.config["m1"]))
        self.seed = int(self.config["seed"])
        self.dataset = self.config["dataset"]
        self.data_dir = self.config["data_dir"]
        self.test_size = float(self.config["test_size"])
        self.file_patterns = self.config["file_patterns"]

    def load_config(self, config_path: str) -> Any:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def generate_synthetic_data(self, custom_synthetic_info: Tuple=None) -> pd.DataFrame:
        """Generate synthetic data according to config."""
        np.random.seed(self.seed)
        n_class, n_feature, n_sample = custom_synthetic_info if custom_synthetic_info else self.synthetic_data_info
        X = np.random.rand(n_sample, n_feature) # 1000 samples and 5 features
        y = np.random.randint(0, n_class, n_sample) # 1000 samples with n classes (class either 0 or 1 -randomized int value)
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_feature)]), pd.Series(y)

    @staticmethod
    def load_real_datasets(dataset_name):
        """Load predefined datasets."""
        if dataset_name == "iris":
            df = load_iris(return_X_y=True)
        elif dataset_name == "digits":
            df = load_digits(return_X_y=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return df
    
    @staticmethod
    def check_folder_files(folderpath: str, patterns: list):
        if not os.path.isdir(folderpath):
            print(f"Folder {folderpath} not found.")
            return False
        for pattern in patterns:
            f_list = glob.glob(os.path.join(folderpath, pattern))
            if not f_list:
                print(f"No matching files found for pattern {pattern}.")
                return False
            
            for f in f_list:
                if os.path.getsize(f) > 0:
                    print(f"File {f} exists and size > 0.")
                else:
                    return False
        return True

    def detect_n_create_new_dataset(self):
        """Check for new datasets & process them."""
        for dataset in self.dataset:
            folder_path = os.path.join(self.data_dir, dataset)
            if not DataPipeline.check_folder_files(folder_path, self.file_patterns):
                print(f"Creating dataset: {dataset}")
                if dataset == "synthetic":
                    X, y = self.generate_synthetic_data()
                else:
                    X, y = DataPipeline.load_real_datasets(dataset)
                X_train, X_test, y_train, y_test = self.preprocess_and_split(X, y)
                self.store_data(X_train, X_test, y_train, y_test, dataset, folder_path)

    def preprocess_and_split(self, X: np.ndarray, y: np.ndarray) -> list:
        """Preprocess features and split dataset."""
        if self.scaler:
            X = self.scaler.fit_transform(X)
        if self.feat:
            X = self.feat.fit_transform(X)
        return train_test_split(X, y, test_size=self.test_size, random_state=self.seed)

    def store_data(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,\
                    dataset_name: str, output_path:str):
        """Store preprocessed dataset."""
        os.makedirs(output_path, exist_ok=True)
        print(f"Storing preprocessed data for Dataset: {dataset_name} ")
        pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1).to_csv(f"{output_path}/train.csv", index=False)
        pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1).to_csv(f"{output_path}/test.csv", index=False)

    def run(self):
        """Run the pipeline for all dataset triggers."""
        self.detect_n_create_new_dataset()
        print("Data Pipeline Execution Completed.")

    def get_data(self) -> list:
        root_dir = self.config["data_pipelines"]["batch"]
        output_dir = root_dir["output_path"]
        data_dir = root_dir["source"]
        csv_f = os.path.join(output_dir, data_dir, "train.csv")
        return self.preprocess_and_split(pd.read_csv(csv_f))

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
