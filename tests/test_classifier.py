import sys
sys.path.append('.')  # Add folder to Python path
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline
import ml_model as ml
import matplotlib.pyplot as plt



from tests.logistic_regression_classification import run_kfold, create_phi_matrix, preprocess_data
from sklearn.model_selection import KFold
import numpy as np
from utils import features as ft
from utils import preprocessing as pp
from sklearn import datasets


import sys
sys.path.append('.')  # Add folder to Python path

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target


lr = ml.LogisticRegressionFromScratch()
lr.fit(X_digits, y_digits)
plt.axvline(lr.best_estimator_.named_steps['gradient_descent'],
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()