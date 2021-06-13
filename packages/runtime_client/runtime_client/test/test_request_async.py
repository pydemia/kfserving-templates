import math
import threading
from timeit import repeat
from runtime_client import KFServingClient, RuntimeClient

import numpy as np
from functools import partial
import aiohttp
import asyncio
import time


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
    infer_func = partial(client.infer_async, MODEL_ID, token=TOKEN)
else:
    client = KFServingClient(DOMAIN)
    infer_func = partial(client.infer_async, MODEL_ID)


def func_np(data, session=None):
    """
    Control function using Numpy.
    """
    return infer_func(data=data, session=session)
    # resp = infer_func(data=data, session=session)
    # if resp.ok:
    #     return resp
    # else:
    #     return ""


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
    return res


DATA = np.random.random(size=(BATCH_SIZE, *DATA_SHAPE))
PARALLEL_DATA = [DATA] * EACH_DATA_NUM


start_time = time.time()
async def main():

    async with aiohttp.ClientSession() as session:

        tasks = []
        for number in range(CONCURRENCY):
            tasks.append(asyncio.ensure_future(func_np(data=DATA, session=session)))

        gathered = await asyncio.gather(*tasks)

    return gathered

gg = asyncio.run(main())

print("--- %s seconds ---" % (time.time() - start_time))

# # correct = timefunc(None, "numpy (1 thread)", func_np, DATA)
# func_nb_mt(arg_zip=zip(PARALLEL_DATA,))
# # func_nb_mt()
# # timefunc(None, "numpy (1 thread)", func_nb_mt, arg_zip=zip(PARALLEL_DATA,))
# # correct = timefunc(None, "numpy (1 thread)", func_nb_mt, arg_zip=zip(PARALLEL_DATA,))
# # multiprocessor(infer_func, worker=1, arg_zip=zip(PARALLEL_DATA,))
# # timefunc(None, "numpy (1 thread)", func_nb)
# timefunc(None, "multiprocess (1 WORKERS)", 
#          multiprocessor, func_np, worker=1, arg_zip=zip(PARALLEL_DATA,))
# timefunc(None, "multiprocess (%d WORKERS)" % WORKERS,
#          multiprocessor, func_np, worker=100, arg_zip=zip(PARALLEL_DATA,))
