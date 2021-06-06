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
import numpy as np
import os
from typing import Union, Dict, List
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
        parsed_request = MultipartRequest(request)
        filename = parsed_request.filename
        instances: np.ndarray = parsed_request.instances

        log.info(filename)

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


class MultipartRequest:
    boundary: bytes
    content_disposition: str
    name: str
    filename: str
    instances: np.ndarray

    def __init__(self, request: bytes):
        parsed_request = [i for i in request.split(b'\r\n') if i]
        if len(parsed_request) == 4:
            (_start_boundary, _content_disposition,
             data_bytes, _end_boundary) = parsed_request
        else:
            (_start_boundary, _content_disposition, _,
             data_bytes, _end_boundary) = parsed_request
        self.boundary = _start_boundary.split(b'--')[-1]
        self.data_bytes = data_bytes
        self.content_disposition, _name, _filename = [
            i.strip().decode('utf8').replace("'", "").replace('"', '')
            for i in _content_disposition.split(b';')]
        self.content_disposition
        self.name = _name.split('name=')[-1]
        self.filename = _filename.split('filename=')[-1]

        try:
            if self.filename.endswith('npz') or self.filename.endswith('npy'):
                npz = np.load(io.BytesIO(data_bytes), allow_pickle=True)
                self.instances = npz[npz.files[0]]
            elif self.filename.endswith('pkl'):
                self.instances = pickle.load(io.BytesIO(data_bytes))
            else:
                npz = np.load(io.BytesIO(data_bytes), allow_pickle=True)
                self.instances = npz[npz.files[0]]
        except Exception as e:
            raise Exception(
                "Unsupported or invalid File Format: " +
                "'npz', 'npy' or 'pkl' format is avaliable.: {}" % e)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.__dict__)
