from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import ml_model as ml
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import accuracy_score

import sys
sys.path.append('.')  # Add folder to Python path


def test_classifier_without_features():
    X, y = load_digits(return_X_y=True)

    # Normalize features for better convergence
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=42)

    labelencoder_y = LabelBinarizer()
    y_train = labelencoder_y.fit_transform(y_train_)
    # Initialize weight matrix
    n_features = X_train.shape[1]  # 64 pixels
    print(f"features {n_features}")
    n_classes = y_train.shape[1]   # 10 digits (0-9)
    print(f"n_class {n_classes}")
    print(f"y_train {y_train.shape}")
    print(f"X_train {X_train.shape}")
    np.random.seed(42)
  
    lr = ml.LogisticRegressionFromScratch(n_class=n_classes)
    lr.fit_transform(X_train, y_train)
    # print(f"original y_train {y_train_}")
    y_pred = lr.predict(X_train)
    res = np.round(accuracy_score(y_train_, y_pred) * 100, 3)
    print(f"Train accuracy score: {res}")

    assert 12.039 == res

    y_pred_ = lr.predict(X_test)
    res_ = np.round(accuracy_score(y_test_, y_pred_) * 100, 3)
    print(f"Test accuracy score: {res_}")

    assert 13.056 == res_
