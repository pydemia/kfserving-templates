# Copyright 2021 .
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kfserving
import joblib
import numpy as np
import os
from typing import Union, Dict
from .utils import change_ndarray_tolist
import logging

import io
import zlib
import pickle

from ..trainer.model import TensorflowModel


log = logging.getLogger(__name__)


class ServingModel(kfserving.KFModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.ready = False

    def load(self) -> bool:
        model_path = kfserving.Storage.download(self.model_dir)

        # Tensorflow:
        self._model = TensorflowModel(
            dirpath=os.path.join(model_path))
        self.ready = True

        return self.ready


    def _predict_from_json(self, request: Dict) -> Dict:
        instances = request["instances"]

        try:
            predictions = self._model.predict(instances).tolist()
            return {
                "predictions": predictions,
                "model_version": os.getenv("MODEL_VERSION", "1")
            }
        except Exception as e:
            raise Exception("Failed to predict %s" % e)

    def _predict_from_bytes(self, request: bytes) -> Dict:
        log.info(type(request))

        (_start_boundary, content_disposition,
         data_bytes, *__) = request.replace(b'\r\n\r\n', b'\r\n').split(b'\r\n')
        *_, filename = content_disposition.split(b';')

        filename = filename.decode('utf8').replace("'", "").replace('"', '')
        if filename.endswith('npz') or filename.endswith('npy'):
            npz = np.load(io.BytesIO(data_bytes), allow_pickle=True)
            instances = npz[npz.files[0]]
        elif filename.endswith('pkl'):
            instances = pickle.load(io.BytesIO(data_bytes))
        else:
            raise Exception("Unsupported or invalid File Format: 'npz', 'npy' or 'pkl' format is avaliable.")

        try:
            predictions = self._model.predict(instances).tolist()
            return {
                "predictions": predictions,
                "model_version": os.getenv("MODEL_VERSION", "1")
            }
        except Exception as e:
            raise Exception("Failed to predict %s" % e)

    def predict(self, request: Union[Dict, bytes]) -> Dict:
        if isinstance(request, dict):
            return self._predict_from_json(request)
        elif isinstance(request, bytes):
            return self._predict_from_bytes(request)
        else:
            raise Exception("Unsupported 'Content-Type': 'json' or 'multipart/form-data is available.")
