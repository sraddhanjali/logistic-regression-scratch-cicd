import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from typing import Union, Sequence
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# for decorator
import functools


import sys
sys.path.append(".")

from config import tuning_params, non_tuning_params

# run on synthetic data
def make_synthetic_data(n_samples: Union[int, Sequence[int]]=100, n_classes: int= 100, random_state: int=100, n_features:int=2):
    """ Make isotropic Gaussian Blobs for synthetic data to work.
    Used for validation of implementation.
    :param n_samples: number of samples in data.
    :param n_classes: number of classes.
    :param n_features: number of features built from descriptors.
    :return: data (ndarray).
    """
    return make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=1.6, random_state=random_state)


# TODO Add this configuration regarding the number of samples and number of class
# TODO via a configuration file like config.yaml

random_state = non_tuning_params["n_random_state"]
n = non_tuning_params["n_samples"]
k_class = non_tuning_params["n_class"]
# the percentage of dataset used for final testing
test_size = non_tuning_params["test_size"]

X, y = make_synthetic_data(n, k_class, random_state)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

from typing import List, Union

def write_to_file(dataset: List):
    for data_ in dataset:
        script_dir, fldname_, fname_, data = data_
        fullpathname = os.path.join(script_dir, fldname_, fname_)
        print(fullpathname)
        if os.path.exists(fullpathname):
            os.remove(fullpathname)
        print(f"Save the pickled training and testing data into the training testing folders")
        print(f"for path: {fullpathname}")
        with open(fullpathname, 'wb') as f:
            pickle.dump(data_, f)

script_dir = sys.path[0]

dataset_info = [[script_dir, "training_data", "train_data.pkl", X_train], 
          [script_dir, "training_data", "train_labels.pkl", y_train],
          [script_dir, "testing_data", "test_data.pkl", X_test],
          [script_dir, "testing_data", "test_labels.pkl", y_test]]


write_to_file(dataset_info)

# run on local data
def read_main_data(file_name):
    data = pd.read_csv(file_name, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    data = shuffle(data)
    labels = data['class']
    descriptors = data.drop(columns=['class'])
    lb = LabelEncoder()
    labels_numerical = lb.fit_transform(labels.values)
    return descriptors.values, labels_numerical, lb


# TODO learn to make an ETL pipeline to load data
local_filepath = os.path.join(script_dir, "../dataset/iris_data.csv")
X, y, lb = read_main_data(local_filepath)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        
iris_dataset_info = [[script_dir, "training_data", "iris_train_data.pkl", X_train], 
          [script_dir, "training_data", "iris_train_labels.pkl", y_train],
          [script_dir, "testing_data", "iris_test_data.pkl", X_test],
          [script_dir, "testing_data", "iris_test_labels.pkl", y_test]]

write_to_file(iris_dataset_info)