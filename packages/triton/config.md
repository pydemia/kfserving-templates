
# tritonserver

```bash
Usage: tritonserver [options]
  --help
        Print usage
  --log-verbose <integer>
        Set verbose logging level. Zero (0) disables verbose logging
        and values >= 1 enable verbose logging.
  --log-info <boolean>
        Enable/disable info-level logging.
  --log-warning <boolean>
        Enable/disable warning-level logging.
  --log-error <boolean>
        Enable/disable error-level logging.
  --id <string>
        Identifier for this server.
  --model-store <string>
        Equivalent to --model-repository.
  --model-repository <string>
        Path to model repository directory. It may be specified
        multiple times to add multiple model repositories. Note that if a model
        is not unique across all model repositories at any time, the model
        will not be available.
  --exit-on-error <boolean>
        Exit the inference server if an error occurs during
        initialization.
  --strict-model-config <boolean>
        If true model configuration files must be provided and all
        required configuration settings must be specified. If false the model
        configuration may be absent or only partially specified and the
        server will attempt to derive the missing required configuration.
  --strict-readiness <boolean>
        If true /v2/health/ready endpoint indicates ready if the
        server is responsive and all models are available. If false
        /v2/health/ready endpoint indicates ready if server is responsive even if
        some/all models are unavailable.
  --allow-http <boolean>
        Allow the server to listen for HTTP requests.
  --http-port <integer>
        The port for the server to listen on for HTTP requests.
  --http-thread-count <integer>
        Number of threads handling HTTP requests.
  --allow-grpc <boolean>
        Allow the server to listen for GRPC requests.
  --grpc-port <integer>
        The port for the server to listen on for GRPC requests.
  --grpc-infer-allocation-pool-size <integer>
        The maximum number of inference request/response objects
        that remain allocated for reuse. As long as the number of in-flight
        requests doesn't exceed this value there will be no
        allocation/deallocation of request/response objects.
  --grpc-use-ssl <boolean>
        Use SSL authentication for GRPC requests. Default is false.
  --grpc-use-ssl-mutual <boolean>
        Use mututal SSL authentication for GRPC requests. Default is
        false.
  --grpc-server-cert <string>
        File holding PEM-encoded server certificate. Ignored unless
        --grpc-use-ssl is true.
  --grpc-server-key <string>
        File holding PEM-encoded server key. Ignored unless
        --grpc-use-ssl is true.
  --grpc-root-cert <string>
        File holding PEM-encoded root certificate. Ignore unless
        --grpc-use-ssl is false.
  --grpc-infer-response-compression-level <string>
        The compression level to be used while returning the infer
        response to the peer. Allowed values are none, low, medium and high.
        By default, compression level is selected as none.
  --allow-metrics <boolean>
        Allow the server to provide prometheus metrics.
  --allow-gpu-metrics <boolean>
        Allow the server to provide GPU metrics. Ignored unless
        --allow-metrics is true.
  --metrics-port <integer>
        The port reporting prometheus metrics.
  --trace-file <string>
        Set the file where trace output will be saved.
  --trace-level <string>
        Set the trace level. OFF to disable tracing, MIN for minimal
        tracing, MAX for maximal tracing. Default is OFF.
  --trace-rate <integer>
        Set the trace sampling rate. Default is 1000.
  --model-control-mode <string>
        Specify the mode for model management. Options are "none",
        "poll" and "explicit". The default is "none". For "none", the server
        will load all models in the model repository(s) at startup and will
        not make any changes to the load models after that. For "poll", the
        server will poll the model repository(s) to detect changes and will
        load/unload models based on those changes. The poll rate is
        controlled by 'repository-poll-secs'. For "explicit", model load and unload
        is initiated by using the model control APIs, and only models
        specified with --load-model will be loaded at startup.
  --repository-poll-secs <integer>
        Interval in seconds between each poll of the model
        repository to check for changes. Valid only when --model-control-mode=poll is
        specified.
  --load-model <string>
        Name of the model to be loaded on server startup. It may be
        specified multiple times to add multiple models. Note that this
        option will only take affect if --model-control-mode=explicit is true.
  --pinned-memory-pool-byte-size <integer>
        The total byte size that can be allocated as pinned system
        memory. If GPU support is enabled, the server will allocate pinned
        system memory to accelerate data transfer between host and devices
        until it exceeds the specified byte size. This option will not affect
        the allocation conducted by the backend frameworks. Default is 256
        MB.
  --cuda-memory-pool-byte-size <<integer>:<integer>>
        The total byte size that can be allocated as CUDA memory for
        the GPU device. If GPU support is enabled, the server will allocate
        CUDA memory to minimize data transfer between host and devices
        until it exceeds the specified byte size. This option will not affect
        the allocation conducted by the backend frameworks. The argument
        should be 2 integers separated by colons in the format <GPU device
        ID>:<pool byte size>. This option can be used multiple times, but only
        once per GPU device. Subsequent uses will overwrite previous uses for
        the same GPU device. Default is 64 MB.
  --min-supported-compute-capability <float>
        The minimum supported CUDA compute capability. GPUs that
        don't support this compute capability will not be used by the server.
  --exit-timeout-secs <integer>
        Timeout (in seconds) when exiting to wait for in-flight
        inferences to finish. After the timeout expires the server exits even if
        inferences are still in flight.
  --backend-directory <string>
        The global directory searched for backend shared libraries.
        Default is '/opt/tritonserver/backends'.
  --repoagent-directory <string>
        The global directory searched for repository agent shared
        libraries. Default is '/opt/tritonserver/repoagents'.
  --buffer-manager-thread-count <integer>
        The number of threads used to accelerate copies and other
        operations required to manage input and output tensor contents.
        Default is 0.
  --backend-config <<string>,<string>=<string>>
        Specify a backend-specific configuration setting. The format
        of this flag is --backend-config=<backend_name>,<setting>=<value>.
        Where <backend_name> is the name of the backend, such as 'tensorrt'.
```

# `config.pbtxt`

`curl localhost:8080/v2/models/<model_name>/config`

```bash
curl localhost:8080/v2/models
curl localhost:18080/v2/models/model/config
```

```bash
saved_model_cli \
  show \
  --dir savedmodel/1/model.savedmodel \
  --tag_set serve \
  --signature_def serving_default

The given SavedModel SignatureDef contains the following input(s):
  inputs['input_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 224, 224, 3)
      name: serving_default_input_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['act_softmax'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1000)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```

```t
# If the name of the model is not specified in the configuration it is assumed to be the same as the model repository directory containing the model.
#name: "simple_identity"

# For TensorRT, 'platform' must be set to tensorrt_plan. Currently, TensorRT backend does not support 'backend' field.
# For PyTorch, 'backend' must be set to pytorch or 'platform' must be set to pytorch_libtorch.
# For ONNX, 'backend' must be set to onnxruntime or 'platform' must be set to onnxruntime_onnx.
# For TensorFlow, 'platform must be set to tensorflow_graphdef or tensorflow_savedmodel. Optionally 'backend' can be set to tensorflow.
# For all other backends, 'backend' must be set to the name of the backend and 'platform' is optional.
platform: "tensorflow_savedmodel"
#backend: "tensorflow"

# max_batch_size should be set to a value greater-or-equal-to 1
max_batch_size: 16

version_policy: { all { }}

input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    dims: [ 224, 224, 3 ]
    is_shape_tensor: false
    # is_shape_tensor: false -> "input_1": [ x, 224, 224, 3 ]
    # is_shape_tensor: true  -> "input_1": [ 224, 224, 3 ]
  }
]
output [
  {
    name: "act_softmax"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    is_shape_tensor: false
  }
]

#instance_group [{ kind: KIND_CPU }]
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
    # gpus: []
    # profile: []
  }
  # {
  #   count: 1
  #   kind: KIND_GPU
  #   gpus: [ 0 ]
  #   # gpus: []
  #   # profile: []
  # },
  # {
  #   count: 2
  #   kind: KIND_GPU
  #   gpus: [ 1, 2 ]
  #   # gpus: []
  #   # profile: []
  # }
]
# Dynamic batching is a feature of Triton that allows inference requests to be combined by the server, so that a batch is created dynamically. Creating a batch of requests typically results in increased throughput. The dynamic batcher should be used for stateless. The dynamically created batches are distributed to all model instances configured for the model.
dynamic_batching {
  preferred_batch_size: [ 2, 4, 8, 16 ]
  # The dynamic batcher can be configured to allow requests to be delayed for a limited time in the scheduler to allow other requests to join the dynamic batch. For example, the following configuration sets the maximum delay time of 100 microseconds for a request.
  max_queue_delay_microseconds: 100
}

batch_input: []
batch_output: []

optimization: {
  priority: PRIORITY_DEFAULT
  input_pinned_memory: {
      enable: true
  }
  output_pinned_memory: {
      enable: true
  }
  gather_kernel_buffer_threshold: 0
  eager_batching: false
}

# parameters [
#   {
#     key: "execute_delay_ms"
#     value: { string_value: "3" }
#   }
# ]
#queue_policy
```
curl --request POST \
  --url http://localhost:8080/v2/models/model/infer \
  --header 'Content-Type: multipart/form-data; boundary=---011000010111000001101001' \
  --header 'ce-contenttype: ' \
  --header 'ce-id: none' \
  --header 'ce-source: none' \
  --header 'ce-specversion: 1.0' \
  --header 'ce-type: none' \
  --form instances=@./input.npz

curl --request POST \
  --url http://localhost:8080/v2/models/model/infer \
  --header 'Content-Type: application/octet-stream' \
  --header 'ce-contenttype: ' \
  --header 'ce-id: none' \
  --header 'ce-source: none' \
  --header 'ce-specversion: 1.0' \
  --header 'ce-type: none' \
  --form instances=@./input.npz

curl --request POST \
  --url http://localhost:8080/v2/models/model/infer \
  --header 'Content-Type: multipart/form-data; boundary=---011000010111000001101001' \
  --header 'ce-contenttype: ' \
  --header 'ce-id: none' \
  --header 'ce-source: none' \
  --header 'ce-specversion: 1.0' \
  --header 'ce-type: none' \
  --form instances=@./input.npz
