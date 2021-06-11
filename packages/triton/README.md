
```bash
# TRITON_VERSION="20.03-py3"
TRITON_VERSION="20.08-py3"
docker pull nvcr.io/nvidia/tritonserver:${TRITON_VERSION}
docker pull nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-min
```


```log

=============================
== Triton Inference Server ==
=============================
NVIDIA Release 20.10 (build <unknown>)
Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying
project or file.
WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use 'nvidia-docker run' to start this container; see
   https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker .
NOTE: The SHMEM allocation limit is set to the default of 64MB.  This may be
   insufficient for the inference server.  NVIDIA recommends the use of the following flags:
   nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ...
E0509 21:57:29.450523 1 pinned_memory_manager.cc:192] failed to allocate pinned system memory: CUDA driver version is insufficient for CUDA runtime version
E0509 21:57:29.450978 1 model_repository_manager.cc:1604] failed to open text file for read /mnt/models/0001/config.pbtxt: No such file or directory
I0509 21:57:29.451042 1 server.cc:141] 
+---------+--------+------+
| Backend | Config | Path |
+---------+--------+------+
+---------+--------+------+
I0509 21:57:29.451059 1 server.cc:184] 
+-------+---------+--------+
| Model | Version | Status |
+-------+---------+--------+
+-------+---------+--------+
I0509 21:57:29.451134 1 tritonserver.cc:1621] 
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                                              |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                                             |
| server_version                   | 2.4.0                                                                                                                                              |
| server_extensions                | classification sequence model_repository schedule_policy model_configuration system_shared_memory cuda_shared_memory binary_tensor_data statistics |
| model_repository_path[0]         | /mnt/models                                                                                                                                        |
| model_control_mode               | MODE_NONE                                                                                                                                          |
| strict_model_config              | 1                                                                                                                                                  |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                                          |
| min_supported_compute_capability | 6.0                                                                                                                                                |
| strict_readiness                 | 1                                                                                                                                                  |
| exit_timeout                     | 30                                                                                                                                                 |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------+
I0509 21:57:29.451150 1 server.cc:280] Waiting for in-flight requests to complete.
I0509 21:57:29.451160 1 server.cc:295] Timeout 30: Found 0 live models and 0 in-flight non-inference requests
error: creating server: Internal - failed to load all models
E0509 21:57:29.451512 1 cuda_memory_manager.cc:65] Failed to finalize CUDA memory manager: [3] CNMEM_STATUS_NOT_INITIALIZED
stream closed
```

## Arguments

```console
$ tritonserver --help
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
  --grpc-server-cert <string>
        File holding PEM-encoded server certificate. Ignored unless
        --grpc-use-ssl is true.
  --grpc-server-key <string>
        File holding PEM-encoded server key. Ignored unless
        --grpc-use-ssl is true.
  --grpc-root-cert <string>
        File holding PEM-encoded root certificate. Ignore unless
        --grpc-use-ssl is false.
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
  --backend-config <<string>,<string>=<string>>
        Specify a backend-specific configuration setting. The format
        of this flag is --backend-config=<backend_name>,<setting>=<value>.
        Where <backend_name> is the name of the backend, such as 'tensorrt'.

For --backend-config for the 'tensorflow' backend the following flags are accepted.
  --backend-config=tensorflow,allow-soft-placement=<boolean>
        Instruct TensorFlow to use CPU implementation of an
        operation when a GPU implementation is not available.
  --backend-config=tensorflow,gpu-memory-fraction=<float>
        Reserve a portion of GPU memory for TensorFlow models.
        Default value 0.0 indicates that TensorFlow should dynamically allocate
        memory as needed. Value of 1.0 indicates that TensorFlow should
        allocate all of GPU memory.
  --backend-config=tensorflow,version=<int>
        Select the version of the TensorFlow library to be used,
        available version is 1 and 2. Default TensorFlow version is 1.
```

## Model File Structure

https://github.com/triton-inference-server/server/blob/829d4ba2d007fdb9ae71d064039f96101868b1d6/docs/model_repository.md#model-files

### Tensorflow

#### GraphDef

```ascii
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.graphdef
```

#### SavedModel

```ascii
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.savedmodel/
           <saved-model files>
```

### TorchScript

```ascii
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.pt
```

### ONNX

### Single File

```ascii
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx
```

```ascii
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx/
           model.onnx
           <other model files>
```

### Python Backend

```ascii
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.py
```

### Build

```bash
docker build . -f tritonserver-base.Dockerfile -t pydemia/tritonserver-base:triton-20.08-py38 > build-tritonserver-base.log 2>&1 && docker push pydemia/tritonserver-base:triton-20.08-py38 \
&& docker save -o pydemia/tritonserver-base:triton-20.08-py38 -o pydemia--tritonserver-base:triton-20.08-py38.tar.gz \
&& gsutil -m cp ./pydemia--tritonserver-base:triton-20.08-py38.tar.gz gs://aiip-runtime-installer/install-offline/pydemia--tritonserver-base:triton-20.08-py38.tar.gz &
```