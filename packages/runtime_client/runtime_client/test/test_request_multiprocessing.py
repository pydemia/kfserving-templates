from itertools import repeat
import math
import threading
from timeit import repeat
from runtime_client import KFServingClient, RuntimeClient

import numpy as np
from functools import partial, wraps
import datetime as dt
import multiprocessing as mpr


import argparse
import logging


log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

parser.add_argument('--client_type', type=str,
                    default='runtime', choices={'kfserving', 'runtime'})
parser.add_argument('--token', type=str, required=False, default=None)
parser.add_argument('--domain', type=str,
                    default="http://localhost:28080", required=False)
parser.add_argument('--model_id', type=str, required=True)
parser.add_argument('--concurrency', type=int, default=1, required=False)
parser.add_argument('--batch_size', type=int, default=1, required=False)
parser.add_argument('--data_shape', type=tuple_type,
                    default=(224, 224, 3), required=False)
parser.add_argument('--each_data_num', type=int, default=20, required=False)


args, _ = parser.parse_known_args()

CLIENT_TYPE = args.client_type
TOKEN = args.token
DOMAIN = args.domain
MODEL_ID = args.model_id
CONCURRENCY = args.concurrency
BATCH_SIZE = args.batch_size
DATA_SHAPE = args.data_shape
EACH_DATA_NUM = args.each_data_num

WORKERS = CONCURRENCY
# size = 10**6

if CLIENT_TYPE == 'runtime':
    client = RuntimeClient(DOMAIN)
    infer_func = partial(client.infer, MODEL_ID, token=TOKEN)
else:
    client = KFServingClient(DOMAIN)
    infer_func = partial(client.infer, MODEL_ID)


def func_np(data):
    """
    Control function using Numpy.
    """
    resp = infer_func(data=data)
    if resp.ok:
        return resp.status_code
    else:
        return 0


def timefunc(correct, s, func, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before the benchmark is
    # started
    res = func(*args, **kwargs)
    # if correct is not None:
    #     assert np.allclose(res, correct), (res, correct)
    # time it
    print('{:>5.0f} ms'.format(min(repeat(
        lambda: func(*args, **kwargs), number=5, repeat=2)) * 1000))
    # return res


# def multiprocessor(func, worker, args_iter, kwargs_iter=None):
#     args_for_starmap = zip(repeat(func), args_iter, kwargs_iter)
#     with mpr.Pool(processes=worker) as pool:
#         resp = pool.starmap(timefunc, args_for_starmap)

#     return resp
def time_profiler(func):

    @wraps(func)
    def profiler(*args, **kwargs):

        start_tm = dt.datetime.now()
        print("(%s) Start   : %26s" % (func.__name__, start_tm))

        res = func(*args, **kwargs)
        end_tm = dt.datetime.now()
        print("(%s) End     : %26s" % (func.__name__, end_tm))

        elapsed_tm = end_tm - start_tm
        print("(%s) Elapsed : %26s" % (func.__name__, elapsed_tm))
        return res

    return profiler


@time_profiler
def multiprocessor(func, worker, arg_zip=None, *args, **kwargs):
    with mpr.Pool(processes=worker) as pool:
        resp = pool.starmap(func, arg_zip, *args, **kwargs)

    return resp

    # with mpr.Pool(processes=worker) as pool:
    #     resp = pool.starmap(func, arg_zip, *args, **kwargs)
    # return resp

# def make_multiprocess(inner_func, numprocesses):
#     def func(arg_zip=None, worker=numprocesses, *args, **kwargs):
#         arg_zip = zip(repeat(func), *args, **kwargs)
#         with mpr.Pool(processes=worker) as pool:
#             pool.starmap(inner_func, arg_zip, *args, **kwargs)
#     return func


DATA = [np.random.random(size=(BATCH_SIZE, *DATA_SHAPE))] * EACH_DATA_NUM * CONCURRENCY
# PARALLEL_DATA = [DATA] * EACH_DATA_NUM


# func_nb = make_multiprocess(func_np, 1)
# func_nb_mt = make_multiprocess(func_np, WORKERS)

# func_nb = partial(multiprocessor, infer_func, 1, arg_zip=zip(PARALLEL_DATA,))
# func_nb_mt = make_multiprocess(func_np, 1)
# func_nb_mt = partial(multiprocessor, infer_func, WORKERS,
#                      arg_zip=zip(PARALLEL_DATA,))

# arg_zip = zip(repeat(infer_func), data=PARALLEL_DATA)
# with mpr.Pool(processes=CONCURRENCY) as pool:
#         pool.starmap(infer_func, *arg_zip)

args_iter = zip(DATA)

# kwargs_iter = repeat(dict(payload={'a': 1}, key=True))
r = multiprocessor(func_np, worker=CONCURRENCY, arg_zip=zip(DATA,))
print(len(r))


# correct = timefunc(None, "numpy (1 thread)", func_np, DATA)
# func_nb_mt(arg_zip=zip(PARALLEL_DATA,))
# func_nb_mt()
# timefunc(None, "numpy (1 thread)", func_nb_mt, arg_zip=zip(PARALLEL_DATA,))
# correct = timefunc(None, "numpy (1 thread)", func_nb_mt, arg_zip=zip(PARALLEL_DATA,))
# multiprocessor(infer_func, worker=1, arg_zip=zip(PARALLEL_DATA,))
# timefunc(None, "numpy (1 thread)", func_nb)
# timefunc(None, "multiprocess (1 WORKERS)", 
#          multiprocessor, func_np, worker=1, arg_zip=zip(PARALLEL_DATA,))
# timefunc(None, "multiprocess (%d WORKERS)" % WORKERS,
#          multiprocessor, func_np, worker=100, arg_zip=zip(PARALLEL_DATA,))
