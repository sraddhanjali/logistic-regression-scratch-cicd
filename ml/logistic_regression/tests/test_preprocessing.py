from sklearn.preprocessing import StandardScaler
import numpy as np

from ml.logistic_regression import preprocessing as pp

import sys
sys.path.append('.')  # Add folder to Python path


X = [[1, 2], [2, 3], [3, 4]]
lib_scaler = StandardScaler()
lib_res = lib_scaler.fit_transform(X)
print(f"Result from lib is : {lib_res}")

scaler = pp.ScalerTransform()
res = scaler.fit_transform(X)
print(f"Result is : {res} and expected: {lib_res}")
np.testing.assert_array_almost_equal(res, lib_res)