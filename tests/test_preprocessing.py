from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

sys.path.append(".")  # Add folder to Python path
from utils import preprocessing as pp
from ml_datapipeline import DataPipeline

def test_preprocessor():
    X = [[1, 2], [2, 3], [3, 4]]
    lib_scaler = StandardScaler()
    lib_res = lib_scaler.fit_transform(X)
    print(f"Result from lib is : {lib_res}")

    scaler = pp.ScalerTransform()
    res = scaler.fit_transform(X)
    print(f"Result is : {res} and expected: {lib_res}")
    np.testing.assert_array_almost_equal(res, lib_res)

def test_processor_with_datapipeline():
    X,_,_,_ = DataPipeline().get_data(dataset='synthetic', debug=True)

    lib_scaler = StandardScaler()
    lib_res = lib_scaler.fit_transform(X)
    print(f"Result from lib is : {lib_res}")

    scaler = pp.ScalerTransform()
    res = scaler.fit_transform(X)
    print(f"Result is : {res} and expected: {lib_res}")
    np.testing.assert_array_almost_equal(res, lib_res)