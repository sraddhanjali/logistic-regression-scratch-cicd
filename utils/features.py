from typing import Tuple, List, Optional, Any
from sklearn.base import OneToOneFeatureMixin, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from itertools import combinations_with_replacement
import numpy as np


class PhiMatrixTransformer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):

    def __init__(self, polynomial_degree: int = 1):
        self.polynomial_degree = polynomial_degree
        self.n_features_out = None
    
    def create_powers_desc(self, X: np.ndarray) -> np.ndarray:
        """ Generate powers of features (polynomial features)."""
        return np.hstack([np.power(X, d) for d in range(1, self.polynomial_degree + 1)])
    
    def create_combination_desc(self, X: np.ndarray) -> np.ndarray:
        """ Generate feature combinations for interaction terms."""
        n_samples, n_features = X.shape
        init_features = []

        for degree in range(1, self.polynomial_degree + 1):
            for combination in combinations_with_replacement(range(n_features), degree):
                new_feature = np.prod([X[:, i] for i in combination], axis=0, keepdims=True)
                init_features.append(new_feature)
        return np.vstack(init_features).reshape(X.shape)
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'PhiMatrixTransformer':
        check_array(X)
        self.is_fitted_ = True
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X).transform(X)
        

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        
        poly_features = self.create_powers_desc(X)
        interaction_features = self.create_combination_desc(X)

        # bias columns of ones
        bias_term = np.ones(X.shape) # number of instances, 1

        print(bias_term.shape, poly_features.shape, interaction_features.shape)
        
        transformed_X =  np.hstack([bias_term, poly_features, interaction_features])
        self.n_features_out = transformed_X.shape
        print(f'The size of the features is {self.n_features_out}')
        return transformed_X
        