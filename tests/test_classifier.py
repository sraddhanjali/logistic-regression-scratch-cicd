import sys
sys.path.append('.')  # Add folder to Python path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import ml_model as ml
import matplotlib.pyplot as plt



from tests.logistic_regression_classification import run_kfold, create_phi_matrix, preprocess_data
from sklearn.model_selection import KFold
import numpy as np
from utils import features as ft
from utils import preprocessing as pp
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import accuracy_score

import sys
sys.path.append('.')  # Add folder to Python path

digits = load_digits()
X, y = digits.data, digits.target  # X: (1797, 64), y: (1797,)

# Normalize features for better convergence
scaler = StandardScaler()
X = scaler.fit_transform(X)



# Train-test split
X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=42)

labelencoder_y = LabelBinarizer()
y_train = labelencoder_y.fit_transform(y_train_)
y_test = labelencoder_y.fit_transform(y_test_)
# Initialize weight matrix
n_features = X_train.shape[1]  # 64 pixels
print(f"features {n_features}")
n_classes = y_train.shape[1]   # 10 digits (0-9)
print(f"n_class {n_classes}")
print(f"y_train {y_train.shape}")
print(f"X_train {X_train.shape}")
np.random.seed(42)
weights = np.random.rand(n_features, n_classes)  # Shape: (64, 10)
# print(f"weights shape {weights.shape}")
# Hyperparameters
learning_rate = 0.1
max_iterations = 1000
reg_lambda = 0.01


lr = ml.LogisticRegressionFromScratch(n_class=n_classes)
lr.fit_transform(X_train, y_train)
# print(f"original y_train {y_train_}")
y_pred = lr.predict(X_train)
print(f"Train accuracy score: {np.round(accuracy_score(y_train_, y_pred) * 100, 3)}")

y_pred_ = lr.predict(X_test)
print(f"Test accuracy score: {np.round(accuracy_score(y_test_, y_pred_) * 100, 3)}")
