import math
import threading
from timeit import repeat
from runtime_client import KFServingClient, RuntimeClient

import numpy as np
from numba import jit

import argparse
import logging


log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--client_type', type=str,
                    default='runtime', choices={'kfserving', 'runtime'})
parser.add_argument('--domain', type=str,
                    default="http://localhost:28080", required=False)
parser.add_argument('--model_id', type=str, required=True)
parser.add_argument('--concurrency', type=int, default=1, required=False)
parser.add_argument('--batch_size', type=int, default=1, required=False)
parser.add_argument('--data_shape', type=tuple, default=(224, 224, 3), required=False)
parser.add_argument('--each_data_num', type=int, default=20, required=False)


args, _ = parser.parse_known_args()

CLIENT_TYPE = args.client_type
DOMAIN = args.domain
MODEL_ID = args.model_id
CONCURRENCY = args.concurrency
BATCH_SIZE = args.batch_size
DATA_SHAPE = args.data_shape
EACH_DATA_NUM = args.each_data_num

nthreads = CONCURRENCY
# size = 10**6

if CLIENT_TYPE == 'runtime':
    client = RuntimeClient(DOMAIN)
else:
    client = KFServingClient(DOMAIN)


def func_np(data):
    """
    Control function using Numpy.
    """
    return client.infer(model_id=MODEL_ID, data=data)

@jit('void(double[:], double[:])', nopython=True,
     nogil=True)
def inner_func_nb(result, data):
    """
    Function under test.
    """
    for i in range(len(result)):
        result[i] = client.infer(model_id=MODEL_ID, data=data[i])


def timefunc(correct, s, func, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before the benchmark is
    # started
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct), (res, correct)
    # time it
    print('{:>5.0f} ms'.format(min(repeat(
        lambda: func(*args, **kwargs), number=5, repeat=2)) * 1000))
    return res


def make_singlethread(inner_func):
    """
    Run the given function inside a single thread.
    """
    def func(*args):
        length = len(args[0])
        result = np.empty(length, dtype=object)
        inner_func(result, *args)
        return result
    return func


def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting
    its arguments into equal-sized chunks.
    """
    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=object)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in
                   args] for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt


func_nb = make_singlethread(inner_func_nb)
func_nb_mt = make_multithread(inner_func_nb, nthreads)


DATA = np.random.random(size=(BATCH_SIZE, *DATA_SHAPE))

correct = timefunc(None, "numpy (1 thread)", func_np, DATA)
timefunc(correct, "numba (1 thread)", func_nb, DATA)
timefunc(correct, "numba (%d threads)" % nthreads, func_nb_mt, DATA)
