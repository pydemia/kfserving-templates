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

from numpy.lib.npyio import NpzFile
import tritonclient.http as httpclient


log = logging.getLogger(__name__)


class ServingModel(kfserving.KFModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.triton_host = f'localhost:{os.getenv("TRITON_HTTP_PORT", "18080")}'
        self.triton_client = None
        self.ready = False

    def load(self) -> bool:
        # model_path = kfserving.Storage.download(self.model_dir)

        if not self.triton_client:
            self.triton_client = httpclient.InferenceServerClient(
                url=self.triton_host, verbose=True)

        self.ready = True
        return self.ready


    def _predict_from_json(self, request: Dict) -> Dict:
        instances = request["instances"]
        inputs, outputs = self.parse_instances(instances)

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

        inputs, outputs = self.parse_instances(instances)

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


    def parse_instances(self, instances: Dict) -> Dict:

        unique_ids = np.zeros([1, 1], dtype=np.int32)
        segment_ids = instances["segment_ids"].reshape(1, 128)
        input_ids = instances["input_ids"].reshape(1, 128)
        input_mask = instances["input_mask"].reshape(1, 128)

        # See: https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md#datatypes
        # FP32, FP16, INT8, INT16, INT32
        inputs = [httpclient.InferInput('unique_ids', [1, 1], "INT32"),
                  httpclient.InferInput('segment_ids', [1, 128], "INT32"),
                  httpclient.InferInput('input_ids', [1, 128], "INT32"),
                  httpclient.InferInput('input_mask', [1, 128], "INT32")]
        inputs[0].set_data_from_numpy(unique_ids)
        inputs[1].set_data_from_numpy(segment_ids)
        inputs[2].set_data_from_numpy(input_ids)
        inputs[3].set_data_from_numpy(input_mask)

        outputs = [httpclient.InferRequestedOutput('start_logits', binary_data=False),
                   httpclient.InferRequestedOutput('end_logits', binary_data=False)]
        return inputs, outputs

    def parse_predictions(self, result_response: Dict) -> Dict:
        end_logits = result_response['outputs'][0]['data']
        start_logits = result_response['outputs'][1]['data']
        return {
            "end_logits": end_logits,
            "start_logits": start_logits,
        }
