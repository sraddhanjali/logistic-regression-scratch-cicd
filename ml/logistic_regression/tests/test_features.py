from sklearn.datasets import make_blobs
import sys
sys.path.append('.')  # Add folder to Python path

from logistic_regression_classification import run_kfold, create_phi_matrix, preprocess_data
from sklearn.model_selection import KFold
import numpy as np
from ml.logistic_regression import features as ft
from ml.logistic_regression import preprocessing as pp

rng = np.random.RandomState(seed=0)  # new random number generator API is recommended

n_samples = 1000
n_classes = 3
n_features = 2
m1 = 1
rate1 = 0.05
max_iters = 500
lamda1 = 0.00001
fold = 2
print("--The code runs for synthetic dataset first and then for iris dataset next--")
print("-----------For m={0} ------".format(m1))
polys = [i for i in range(1, m1 + 1)]
print("-----------FOR SYNTHETIC DATASET---------")
synth_data  = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=1.6, random_state=rng)
kf = KFold(n_splits=fold, shuffle=True, random_state=rng)
kf.get_n_splits(synth_data[0])
from sklearn.preprocessing import StandardScaler

def printer(x):
    try:
        print(f"X={x},type={type(x)}, shape={x.shape}")
    except Exception as e:
        print(f"Exception as e: {e}")
    finally:
        print(f"X={x},type={type(x)}, shape={x.shape}")

for train_ind, test_ind in kf.split(synth_data[0][:, 0]):
    X_train = []
    for i in range(synth_data[0].shape[1]):
        X_train.append(synth_data[0][:, i][train_ind])
    lib_scaler = StandardScaler()
    lib_res = lib_scaler.fit_transform(X_train)
    printer(lib_res)
    
    #### test for new implementation
    scaler = pp.ScalerTransform()
    res = scaler.fit_transform(X_train)  
    printer(res)
    np.testing.assert_array_almost_equal(lib_res, res, err_msg="The Preprocess logic failed. Check ScalerTransform")


    phi_fin_train = create_phi_matrix(lib_res, lib_res.shape, polys)
    printer(phi_fin_train)

    phi = ft.PhiMatrixTransformer(polynomial_degree=m1)
    phimatrix = phi.fit_transform(res)
    printer(phimatrix)
    np.testing.assert_array_almost_equal(phi_fin_train, phimatrix, err_msg="The phi matrix transformer is not working as expected. Check PhiMatrixTransformer")
    break
