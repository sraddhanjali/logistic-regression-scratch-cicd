from typing import Tuple, List, Optional

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class ATransform(BaseEstimator, TransformerMixin): # python's MRO (method resolution order) is from left to right & Transformermixin might use core functionality from BaseEstimator

    # def __init__(self, base: int = 0):
    def fit(self, X: np.ndarray) -> 'ATransform':
        # nothing to learn here from X
        self.mean = np.mean(X) # create an instance variable mean only used for this instance 
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return np.multiply(np.asarray(X), self.mean)


class BTransform(BaseEstimator, TransformerMixin): # python's MRO (method resolution order) is from left to right & Transformermixin might use core functionality from BaseEstimator

    # def __init__(self, base: int = 0):

    def fit(self, X: np.ndarray) -> 'BTransform':
        # nothing to learn here from X
        self.mean = 3 # create an instance variable mean only used for this instance 
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return np.multiply(np.asarray(X), self.mean)


class VerbosePipeline(Pipeline):

    def _fit_transform(self, X: np.ndarray):
        for (name, step) in self.steps:
            print(f"Name of function {name}, Type {type(name)}")
            inst = step().fit(X)
            print(f"Multiplier used in step {name}: {inst.__getattribute__('mean')}")
            X = inst.transform(X)
            print(f"After {name}: ", X)
        return X

    def fit_transform(self, X: np.ndarray, y=None, **fit_params):
        return self._fit_transform(X)

if __name__ == "__main__":
    input_d = [1, 2, 4, 5, 6]
    pipeline =  Pipeline([
    ('atransform', ATransform()), 
    ('btransform', BTransform())
    ])

    print(f" Input before {input_d}")
    print(f" Non-verbose pipeline {pipeline.fit_transform(input_d)} output")
    print(f" Input after pipeline {input_d}")
    
    verbose_pipeline = VerbosePipeline([
        ('atransform', ATransform), 
        ('btransform', BTransform)
    ])

    print(f" Input after verbose pipeline {input_d}")
    print(f" Non-verbose pipeline {verbose_pipeline.fit_transform(input_d)} output")
