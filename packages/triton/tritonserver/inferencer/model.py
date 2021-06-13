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
from typing import Union, Dict, List
from .utils import change_ndarray_tolist
import logging

import re
import io
import zlib
import pickle

from numpy.lib.npyio import NpzFile
import tritonclient.http as httpclient


log = logging.getLogger(__name__)


class ServingModel(kfserving.KFModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.triton_host = f'localhost:{os.getenv("TRITON_HTTP_PORT", "8000")}'
        self.triton_client = None
        self.ready = False

    def load(self) -> bool:
        # model_path = kfserving.Storage.download(self.model_dir)

        if not self.triton_client:
            self.triton_client = httpclient.InferenceServerClient(
                url=self.triton_host, verbose=False)

        self.ready = True
        return self.ready


    def _predict_from_json(self, request: Dict) -> Dict:
        instances = np.array(request["instances"])
        inputs, outputs = self.parse_instances_from_ndarray(instances)

        try:
            # result = self._model.predict(inputs).tolist()
            result = self.triton_client.infer(self.name, inputs=inputs, outputs=outputs)
            predictions = self.parse_predictions(result.get_response())
            return {
                "predictions": predictions,
                "model_version": os.getenv("MODEL_VERSION", "1")
            }
        except Exception as e:
            raise Exception("Failed to predict %s" % e)

    def _predict_from_bytes(self, request: bytes) -> Dict:
        parsed_request = MultipartRequest(request)
        # filename = parsed_request.filename
        instances: np.ndarray = parsed_request.instances

        inputs, outputs = self.parse_instances_from_ndarray(instances)

        try:
            # result = self._model.predict(inputs).tolist()
            result = self.triton_client.infer(self.name, inputs=inputs, outputs=outputs)
            predictions = self.parse_predictions(result.get_response())
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


    # def parse_instances_from_list(self, instances: List) -> Dict:

    #     # _input = instances["input_1"].reshape(-1, 224, 224, 3)
    #     # _input = instances.get("input_1", ).reshape(-1, 224, 224, 3).astype(np.float32)
    #     instances.reshape(-1, 224, 224, 3).astype(np.float32)

    #     # See: https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#datatypes
    #     # FP32, FP16, INT8, INT16, INT32
    #     inputs = [httpclient.InferInput('input_1', _input.shape, "FP32")]
    #     inputs[0].set_data_from_numpy(_input)

    #     outputs = [httpclient.InferRequestedOutput('act_softmax', binary_data=False)]
    #     return inputs, outputs

    def parse_instances_from_ndarray(self, instances: np.ndarray) -> Dict:

        # _input = instances["input_1"].reshape(-1, 224, 224, 3)
        _input = instances.reshape(-1, 224, 224, 3).astype(np.float32)

        # See: https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#datatypes
        # FP32, FP16, INT8, INT16, INT32
        inputs = [httpclient.InferInput('input_1', _input.shape, "FP32")]
        inputs[0].set_data_from_numpy(_input)

        outputs = [httpclient.InferRequestedOutput(
            'act_softmax', binary_data=False)]
        return inputs, outputs

    def parse_predictions(self, result_response: Dict) -> Dict:
        _logits = result_response['outputs'][0]['data']
        return {
            "_logits": _logits
        }


class MultipartRequest:
    # boundary: bytes
    # content_disposition: str
    # content_type: str
    # name: str
    # filename: str
    data_bytes: bytes
    instances: np.ndarray

    def __init__(self, request: bytes):
        _info, data_bytes = request.split(b'\r\n\r\n')[:2]
        #parsed_request = [i for i in request.split(b'\r\n') if i]
        #if len(parsed_request) == 4:
        #    (_start_boundary, _content_disposition,
        #     data_bytes, _end_boundary) = parsed_request
        #else:
        #    (_start_boundary, _content_disposition, _,
        #    data_bytes, _end_boundary) = parsed_request

        # _info = _info.decode()
        # matcher = re.compile(r'Content-Type: (?P<content_type>(.+))\r\n')
        # matched = matcher.search(_info)

        # _content_type = re.findall(r'Content-Type: (.+)\r\n', _info)
        # self._content_type = _content_type if _content_type else None
        # _content_disposition = re.findall(
        #     r'Content-Disposition: (.+)\r\n', _info)
        # self._content_disposition = _content_disposition[0] if _content_disposition else None

        self.data_bytes = data_bytes
        # _disp = _content_disposition.replace("'", "").replace('"', '')
        # _disp = [
        #     i.strip().replace("'", "").replace('"', '')
        #     for i in _content_disposition.split(';')]
        # self.name = [i.split("name=")[-1] for i in _disp if i.startswith("name=")]
        # self.filename = [i.split("filename=")[-1]
        #                  for i in _disp if i.startswith("filename=")]

        try:
            npz = np.load(io.BytesIO(data_bytes), allow_pickle=True)
            self.instances = npz[npz.files[0]]
            # if self.filename.endswith('npz') or self.filename.endswith('npy'):
            #     npz = np.load(io.BytesIO(data_bytes), allow_pickle=True)
            #     self.instances = npz[npz.files[0]]
            # elif self.filename.endswith('pkl'):
            #     self.instances = pickle.load(io.BytesIO(data_bytes))
            # else:
            #     npz = np.load(io.BytesIO(data_bytes), allow_pickle=True)
            #     self.instances = npz[npz.files[0]]
        except Exception as e:
            raise Exception(
                "Unsupported or invalid File Format: " +
                "'npz', 'npy' or 'pkl' format is avaliable.: {}" % e)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.__dict__)
