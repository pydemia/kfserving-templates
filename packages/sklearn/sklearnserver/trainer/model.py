
"""
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ._core import BaseModel
from .preprocessor import prep_func
from .postprocessor import post_func
from . import config

import os
import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler # StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import ensemble


__all__ = ['SKLearnModel']


class SKLearnModel(BaseModel):

    def __init__(self, filepath=None, *args, **kwargs):
        super().__init__(filepath=filepath, *args, **kwargs)
        self.config = config

    def build(self, penalty='l2'):
        # sc = StandardScaler()
        # sc = MinMaxScaler()
        pca = PCA()
        logistic = LogisticRegression(
            penalty=penalty,
            max_iter=10000,
            tol=0.1,
        )
        pipe = Pipeline(
            [
                # ('scaler', sc),
                ('decomp', pca),
                ('regression', logistic),
            ]
        )
        self.model = pipe

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        # Case 1: `prep_func` is made for the batch `X`
        X = prep_func(X)
        # Case 2: `prep_func` is made for each input in `X`
        X = np.array([prep_func(x) for x in X])
        return X

    def postprocess(self, y_hat: np.ndarray) -> np.ndarray:
        # Case 1: `post_func` is made for the batch `y_hat`
        y_hat = post_func(y_hat)
        # Case 2: `post_func` is made for each output in `y_hat`
        y_hat = np.array([post_func(yh) for yh in y_hat])
        return y_hat

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self.preprocess(X), y)

    def evaluate(self, X: np.ndarray, y: np.ndarray = None, sample_weight=None, *args, **kwargs):
        return self.model.score(X, y, sample_weight=sample_weight, *args, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.postprocess(self.model.predict(self.preprocess(X)))

    def save(self, filepath, *args, **kwargs):
        os.makedirs(filepath, exist_ok=True)
        joblib.dump(self.model, os.path.join(filepath, 'model.joblib'), *args, **kwargs)

    def load(self, filepath, *args, **kwargs):
        self.model = joblib.load(os.path.join(filepath, 'model.joblib'), *args, **kwargs)
