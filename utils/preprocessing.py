from typing import Tuple, List, Optional

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, check_array

from sklearn.preprocessing import StandardScaler


class ScalerTransform(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """A custom scaler transform for custom logistic regression classifier implementation.

        This transform defines the following functionality: 

        a fit_transform method that delegates to fit and transform;

        if mean_ and std_ are defined, then ScalerTransform will use it to scale X.
        
        Examples
        >>> transformer = ScalerTransform()
        >>> X = [[1, 2], [2, 3], [3, 4]]
        >>> transformer.fit_transform(X)
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    # Pipeline expects data through fit and transform
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        return self.fit(X).transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ScalerTransform':
        try:
            X = check_array(X)
        except AttributeError as e:
            raise AttributeError
        finally:
            self.mean_ = np.mean(X, axis=0) # column-wise for features to be scaled independently of other features
            self.std_ = np.std(X, axis=0)
            # to prevent division by zero
            self.std_[self.std_ == 0] = 1e-8
            self.is_fitted_ = True        
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        check_is_fitted(self)
        X = check_array(X)
        return (X - self.mean_) / self.std_
    
