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

import argparse
import logging
import sys

import kfserving
from .model import ServingModel
from .model_repository import ModelRepository

DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/models"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument(
    '--model_dir',
    required=True,
    help='A URI pointer to the model binary',
)
parser.add_argument(
    '--model_name',
    required=True,
    default=DEFAULT_MODEL_NAME,
    help='The name that the model is served under.',
)
# User Definition ============================================================

DEFAULT_VALUE = ""
parser.add_argument(
    '--user_defined_arg',
    default=DEFAULT_VALUE,
    help='An user defined arguments for inference.',
)

# ============================================================================

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = ServingModel(args.model_name, args.model_dir)
    try:
        model.load()
    except Exception as e:
        ex_type, ex_value, _ = sys.exc_info()
        logging.error(f"fail to load model {args.model_name} from dir {args.model_dir}. "
                      f"exception type {ex_type}, exception msg: {ex_value}")
        model.ready = False

    server = kfserving.KFServer(registered_models=ModelRepository(args.model_dir))
    server.start([model] if model.ready else [])
