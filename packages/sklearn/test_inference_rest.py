#!/usr/bin/env python

# %%

import os
import ssl
import json
import urllib
import requests
import tqdm

import argparse
import numpy as np

# ssl._create_default_https_context = ssl._create_unverified_context

new_x = np.random.rand(1, 10)
test_input_payload = {"instances": new_x}


def request(
        _input,
        host=None,
        model_name=None, op='predict', version='v2',
        ):

    if (isinstance(_input, str) and os.path.splitext(_input)[1] == 'json'):
        with open(_input, 'r') as f:
            payload = json.load(f)
    elif (isinstance(_input, dict) and 'instances' in _input):
        if (isinstance(_input["instances"], np.ndarray)):
            payload = {
                "instances": _input["instances"].tolist(),
            }
        else:
            payload = _input
    else:
        payload = {
            "instances": _input,
        }
    with open('input.json', 'w') as f:
        json.dump(payload, f)

    RESTAPI_TEMPLATE = 'http://{0}/v1/models/{1}:{2}'
    url = RESTAPI_TEMPLATE.format(host, model_name, op)
    if version == 'v2':
        RESTAPI_TEMPLATE = 'http://{0}/v2/models/{1}/infer'
        url = RESTAPI_TEMPLATE.format(host, model_name)

    hostname = f'{MODEL_NAME}.{NAMESPACE}.example.com'
    headers = {'Host': hostname}

    print(f"\nCalling '{url}'\nHeader: {headers}")
    print(f'\nBody: \n{payload}')
    response = requests.post(url, json=payload, headers=headers)

    return response, op


def save_raw_response(response, op):

    SAVEPATH = 'output.json'
    if response.status_code == 200:
        resp = json.loads(response.content.decode('utf-8'))

        with open(SAVEPATH, 'w') as f:
            json.dump(resp, f)
        print("\nop={}: result saved [{}]".format(op, SAVEPATH))

    else:
        print(f"Received response: \ncode: {response.status_code, response.reason}\ncontent: {response.content}")


def predict(
        _input,
        host=None, hostname=None, model_name=None, version='v2',
        ):
    response, op = request(
        _input=_input,
        host=host,
        model_name=model_name,
        op='predict',
        version=version,
    )
    save_raw_response(response, op)
    print(f'\nBody: \n{response.content}')


def explain(
        _input,
        host=None, hostname=None, model_name=None, version='v2',
        ):
    resp_pred, _ = request(
        _input=_input,
        host=host,
        model_name=model_name,
        op='predict',
        version=version,
    )
    # resp = json.loads(resp_pred.content.decode('utf-8'))
    # # label = resp['pred_decode']
    # label = decode_predictions(resp, top=1)
    # print(label)
    # response, op = request(
    #     _input=_input,
    #     host=host,
    #     hostname=hostname,
    #     model_name=model_name,
    #     op='explain',
    # )
    # # decode_response(response, op)
    # save_raw_response(response, op)
    print(f'\nBody: \n{resp_pred.content}')


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=test_input_payload,
                    help='URL of Istio Ingress Gateway')
parser.add_argument('--host', default=os.environ.get("SERVER_HOST"),
                    help='URL of Istio Ingress Gateway')
parser.add_argument('--namespace', default=os.environ.get("NAMESPACE"),
                    help='Namespace where the Inferenceservice is running')
parser.add_argument('--model_name', default=os.environ.get("MODEL_NAME"),
                    help='MODEL_NAME')
parser.add_argument('--op', choices=["predict", "explain"], default="predict",
                    help='Operation to run')
parser.add_argument('--version', choices=["v1", "v2"], default="v2",
                    help='KFServing API version.')
args, _ = parser.parse_known_args()


TEST_INPUT = args.input
OP = args.op
host = args.host
NAMESPACE = args.namespace
MODEL_NAME = args.model_name
API_VERSION = args.version

if __name__ == "__main__":

    if OP == "predict":
        predict(
            _input=TEST_INPUT,
            host=host,
            model_name=MODEL_NAME,
            version=API_VERSION,
        )
    elif OP == "explain":
        explain(
            _input=TEST_INPUT,
            host=host,
            model_name=MODEL_NAME,
            version=API_VERSION,
        )

"""
CLUSTER_HOST="localhost:8080"
MODEL_NAME="sklearnserver-test"
NAMESPACE="default"

python test_inference_rest.py \
    --host $CLUSTER_HOST \
    --model_name $MODEL_NAME \
    --namespace $NAMESPACE \
    --op predict

python test_inference_rest.py \
    --host $CLUSTER_HOST \
    --model_name $MODEL_NAME \
    --namespace $NAMESPACE \
    --op explain
"""


"""
CLUSTER_HOST=""
MODEL_NAME="sklearnserver-test"
NAMESPACE="test"

python test_inference_rest.py \
    --host $CLUSTER_HOST \
    --model_name $MODEL_NAME \
    --namespace $NAMESPACE \
    --op predict

python test_inference_rest.py \
    --input input.json \
    --host $CLUSTER_HOST \
    --model_name $MODEL_NAME \
    --namespace $NAMESPACE \
    --op predict

python test_inference_rest.py \
    --input input_numpy_tensorname.json \
    --host $CLUSTER_HOST \
    --model_name $MODEL_NAME \
    --namespace $NAMESPACE \
    --op predict

python test_inference_rest.py \
    --input input_numpy_multi_images.json \
    --host $CLUSTER_HOST \
    --model_name $MODEL_NAME \
    --namespace $NAMESPACE \
    --op predict

python test_inference_rest.py \
    --input input_numpy_multi_images_tensorname.json \
    --host $CLUSTER_HOST \
    --model_name $MODEL_NAME \
    --namespace $NAMESPACE \
    --op predict

"""
# %%
