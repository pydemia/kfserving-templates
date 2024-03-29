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
import pandas as pd
import json
import os
from typing import Dict

from ..trainer.model import SKLearnModel


class ServingModel(kfserving.KFModel):  # pylint:disable=c-extension-no-member
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.ready = False

    def load(self) -> bool:
        model_path = kfserving.Storage.download(self.model_dir)
        
        # Scikit-learn: joblib, pkl, pickle
        # paths = [os.path.join(model_path, MODEL_BASENAME + model_extension)
        #          for model_extension in MODEL_EXTENSIONS]
        # for path in paths:
        #     if os.path.exists(path):

        #         self._model = joblib.load(path)

        #         self.ready = True
        #         break
        self._model = SKLearnModel(
            dirpath=os.path.join(model_path))

        # Tensorflow:
        model_num = '0001'
        # self._model = trainer.Model(dirpath=os.path.join(model_path, model_num))
        self.ready = True

        return self.ready

    # def user_model(self)

    def predict(self, request: Dict) -> Dict:
        instances = request["instances"]
        labels = request["labels"]
        datatypes = request["types"]
        try:
            instance_t = [list(x) for x in zip(*instances)] # transpose 
            X_inputs = pd.DataFrame(index=None)
            types = [
                'object' if t == 'text' or t == 'datatime64' 
                else t 
                for t in datatypes
            ]
            for c, v, d in zip(labels, instance_t, types):
                X_inputs[c] = pd.Series(v, dtype=d)

        except Exception as e:
            raise Exception(
                "Failed to initialize NumPy array from inputs: %s, %s" % (e, instances))
        try:
            result = self._model.predict(X_inputs).tolist()
            return {"predictions": result}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)
