import os
import io
import sys
import json
import zlib
import pickle
import numpy as np
from tempfile import TemporaryFile

import requests

with open("input.json", "r") as f:
    original = json.load(f)

with open("input-large.json", "r") as f:
    original = json.load(f)

# npz


ndarray = np.array(original["instances"])
np.savez_compressed('input-large.npz', instances=ndarray)

loaded = np.load('input-large.npz')
ndarray_loaded = loaded[loaded.files[0]]  # loaded['instances']

print(f"Is two arrays Identical? \n[{np.array_equal(ndarray, ndarray_loaded)}]\n")  # True
print(f"ndarray: {sys.getsizeof(ndarray) / 1024} == {sys.getsizeof(ndarray_loaded) / 1024}")
print(f"File Size[npz]: {os.path.getsize('input-large.npz') / 1024}")

npz_as_bytes = TemporaryFile()
np.savez_compressed(npz_as_bytes, instances=ndarray)

def save_np_as_filestream(ndarray: np.ndarray):
    npz_as_bytes = TemporaryFile()
    np.savez_compressed(npz_as_bytes, instances=ndarray)
    npz_as_bytes.seek(0)  # Only needed here to simulate closing & reopening file
    return npz_as_bytes


npz_as_bytes.seek(0) # Only needed here to simulate closing & reopening file
sys.getsizeof(npz_as_bytes)
print(f"ndarray: {sys.getsizeof(ndarray) / 1024} -> {sys.getsizeof(npz_as_bytes) / 1024}")
ndarray_byte_loaded = np.load(npz_as_bytes, allow_pickle=True)
ndarray_byte_loaded[ndarray_byte_loaded.files[0]]  # ndarray_byte_loaded['instances']

# zlib

with open("input-large.zlib", "wb") as f:
    ndarray_zcomp = zlib.compress(ndarray)
    f.write(ndarray_zcomp)

with open("input-large.zlib", "rb") as f:
    ndarray_zcomp_loaded = f.read()

ndarray_decomp = zlib.decompress(ndarray_zcomp_loaded)

print(f"zlib-compressed: {sys.getsizeof(ndarray_zcomp) / 1024}")
print(f"File Size[zlib]: {os.path.getsize('input-large.zlib') / 1024}")
print(f"zlib-compressed_loded: {sys.getsizeof(ndarray_zcomp_loaded) / 1024}")
print(f"zlib-decompressed: {sys.getsizeof(ndarray_decomp) / 1024}")
print(f"zlib: {sys.getsizeof(ndarray_zcomp) / 1024}, {sys.getsizeof(ndarray_decomp) / 1024}")

print(f"Is two arrays Identical? \n[{np.array_equal(ndarray, ndarray_loaded)}]\n")  # True
print(f"ndarray: {sys.getsizeof(ndarray) / 1024} == {sys.getsizeof(ndarray_decomp) / 1024}")
print(f"File Size[npz]: {os.path.getsize('input-large.npz') / 1024}")


# pickle

with open("input-large.pkl", "wb") as f:
    pickle.dump(ndarray, f)

with open("input-large.pkl", "rb") as f:
    ndarray_pkl_loaded = pickle.load(f)

print(f"Is two arrays Identical? \n[{np.array_equal(ndarray, ndarray_pkl_loaded)}]\n")  # True
print(f"ndarray: {sys.getsizeof(ndarray) / 1024}, {sys.getsizeof(ndarray_pkl_loaded) / 1024}")
print(f"File Size[pkl]: {os.path.getsize('input-large.pkl') / 1024}")


# requests.post

url = "http://localhost:8080/v2/models/test/infer"
header = {
    "ce-specversion": "1.0",
    "ce-source": "none",
    "ce-type": "none",
    "ce-id": "none",
    "Content-Type": "multipart/form-data",
}
filename = "instances.npz"
files = {
    "instances": (filename, save_np_as_filestream(ndarray))
}
requests.post(url, headers=header, files=files)


filename = "instances.pkl"
files = {
    "instances": (filename, ndarray)
}
requests.post(url, headers=header, files=files)

# FILE #############################################
with open("input.json", "r") as f:
    original = json.load(f)

ndarray = np.array(original["instances"])

url = "http://localhost:8080/v2/models/test/infer"
header = {
    "ce-specversion": "1.0",
    "ce-source": "none",
    "ce-type": "none",
    "ce-id": "none",
    "Content-Type": "multipart/form-data",
}
filename = "instances.npz"
files = {
    "instances": (filename, save_np_as_filestream(ndarray))
}
requests.post(url, headers=header, files=files).json()

# JSON #############################################

url = "http://localhost:8080/v2/models/test/infer"
header = {
    "Content-Type": "application/json",
}
body = {
    "instances": ndarray.tolist()
}
requests.post(url, headers=header, json=body).json()
