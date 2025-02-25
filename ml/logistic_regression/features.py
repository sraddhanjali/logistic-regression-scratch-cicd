from typing import Tuple, List, Optional, Any
from sklearn.base import OneToOneFeatureMixin, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from itertools import combinations
import numpy as np


class PhiMatrixTransformer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):

    def __init__(self, polynomial_degree: int = 1):
        self.complexities = [i for i in range(1, polynomial_degree + 1)]
    
    def create_powers_desc(self, X: np.ndarray) -> Any:
        """ Adds powers of descriptors using values in polys.
        :param descriptors: a list of descriptors.
        :param complexities: a list of complexities.
        :return: iterable for powers.
        """
        for degree in self.complexities[1:]:
            for d in X:
                r = np.power(d, degree)
                yield r.T
    
    def create_combination_desc(self, X: np.ndarray) -> Any:
        """ Build combinations of descriptors.
            :param descriptors: a list of descriptors.
            :param complexities: a list of polynomial complexities.
            :param d_shape: number of instances in data.
            :return: iterable for combination of descriptors.
        """
        for degree in self.complexities:
            indices = [i for i in range(len(X))]
            iterab = combinations(indices, degree)
            for it in iterab:
                mult = np.ones(X.shape)
                try:
                    check_mult = mult*X[0]
                except ValueError:
                    mult = mult.T
                for i in it:
                    mult *= X[i]
                yield mult.T

    def create_phi_matrix(self, X: np.ndarray, d_shape: np.ndarray) -> np.matrix[Any]:
        """ Build a phi data from descriptors.
        :param descriptors: a list of descriptors.
        :param d_shape: number of instances in data.
        :param complexities: a list of polynomial complexities.
        :return: matrix of features built from desc as a list.
        """
        phi = []
        one = np.ones(d_shape)
        phi.append(one)
        for val in self.create_combination_desc(X):
            phi.append(val)
        for val in self.create_powers_desc(X):
            phi.append(val)
        print(f"phi test {type(phi)}, {len(phi)}")
        try:
            return np.matrix(np.vstack(phi)).T
        except ValueError:
            return np.matrix(np.vstack(phi.T)).T
    
    def fit(self, X: np.ndarray) -> 'PhiMatrixTransformer':
        try:
            check_array(X)
        except AttributeError as e:
            raise AttributeError
        finally:
            self.is_fitted_ = True
            return self

    def transform(self, X: np.ndarray) -> np.matrix:
        try:
            X = check_array(X)
        except AttributeError as e:
            raise AttributeError
        finally:
            check_is_fitted(self)
            return self.create_phi_matrix(X.T, X.shape)
        