# pysparkserver

## Architecture

```ascii
Project
├── Makefile
├── README.md
├── model.Dockerfile  <-------------------------- Dockerfile for Build
├── modelserver
│   ├── __init__.py
│   ├── inferencer
│   └── trainer
├── run_modelserver   <-------------------------- Container Entrypoint
└── setup.py
```

* `modelserver`

```ascii
${framework}server
├── example_models/0001 <------------------------ Pre-built model files as sample
├── __init__.py
├── inferencer        <-------------------------- Model Inference with API
│   ├── __init__.py
│   ├── __main__.py   <-------------------------- Runnable for Inference API Server
│   ├── model.py      <-------------------------- (inherited from KFserving.Model)
│   ├── model_repository.py  <------------------- Model Loading for Inference
│   ├── test_model.py
│   └── test_model_repository.py
└── trainer           <-------------------------- Model Definition & Training
    ├── 0001
    ├── __init__.py
    ├── __main__.py   <-------------------------- Runnable for Training(Optional)
    ├── config.py       <------------------------ Env. Variables & Config.
    ├── model.py      <-------------------------- Runnable for Training
    ├── postprocessor.py  <---------------------- Postprocessing Module
    ├── preprocessor.py   <---------------------- Preprocessing Module
    └── utils.py         <----------------------- Utility Module
```

---

## Quickstart

* Build `pysparkserver.Dockerfile`

```bash
cd sklearn
docker build -f ./pysparkserver.Dockerfile -t pysparkserver:0.1.0 .
```

```bash
docker run --rm \
  --mount type=bind,source="$(pwd)/example_models/0001/",target=/mnt/models \
  -e MODEL_NAME=pysparkserver-test \
  -p 8080:8080 \
  pysparkserver:0.1.0
```

```bash
CLUSTER_HOST="localhost:8080"
MODEL_NAME="pysparkserver-test"
NAMESPACE="default"

python test_inference_rest.py \
    --host $CLUSTER_HOST \
    --model_name $MODEL_NAME \
    --namespace $NAMESPACE \
    --op predict
```



The output would be the following:

* `test_inference_rest.py`:

```ascii
Calling 'http://localhost:8080/v2/models/pysparkserver-test/infer'
Header: {'Host': 'http://pysparkserver-test.default.example.com'}

{'instances': [[0.982474111951369, 0.08300793829000164, 0.53042101721823, 0.44908077567432925, 0.7338050972635636, 0.7168371094530415, 0.58404831145148, 0.9296786945364722, 0.49658817714538406, 0.6259569415834348]]}
predict: result saved: output.json
b'{"predictions": [0]}'
```

or use `curl`:

```bash
$ curl -X POST "http://localhost:8080/v2/models/pysparkserver-test/infer" \
    -H "Host: http://pysparkserver-test.default.example.com" \
    -H 'Content-Type: application/json' \
    -d '@./input.json'

{"predictions": [0]}
```
