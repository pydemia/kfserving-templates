
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
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler # StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import ensemble


__all__ = ['SKLearnModel']


class SKLearnModel(BaseModel):

    def __init__(self, dirpath=None, *args, **kwargs):
        self.le = None # label encoder (le.pkl 파일이 있을 경우만 사용)
        super().__init__(dirpath=dirpath, *args, **kwargs)
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

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return prep_func(X)

    def postprocess(self, y_hat):   # 분류 문제에서 label encoder 사용시에 동작
        if self.le: 
            try:
                y_hat = self.le.inverse_transform(y_hat)
            except Exception as e:
                pass

        return y_hat

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(self.preprocess(X), y)

    def evaluate(self, X: np.ndarray, y: np.ndarray = None, sample_weight=None, *args, **kwargs):
        return self.model.score(X, y, sample_weight=sample_weight, *args, **kwargs)

    def predict(self, X: np.ndarray):
        return self.postprocess(self.model.predict(X))

    def save(self, dirpath, *args, **kwargs):
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.model, os.path.join(dirpath, 'model.joblib'), *args, **kwargs)

    def load(self, dirpath, *args, **kwargs):
        self.model = joblib.load(os.path.join(dirpath, 'model.joblib'), *args, **kwargs)

        lepath = os.path.join(dirpath, 'le.pkl')
        try:
            with open(lepath, 'rb') as f:
                self.le = pickle.load(f)
        except IOError as e:
            pass
