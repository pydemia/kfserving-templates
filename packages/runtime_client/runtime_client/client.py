import os
import io
import sys
import json
import zlib
import pickle
import dill
import numpy as np
from tempfile import TemporaryFile

from typing import Coroutine, DefaultDict, Union, Dict
from collections import namedtuple
from dataclasses import dataclass
import requests
import aiohttp
from requests_toolbelt import MultipartEncoder, MultipartDecoder


__all__ = [
    "KFServingClient",
    "RuntimeClient",
]

@dataclass
class KFServingUrlFormat:
    live: str = "{domain}/health/live"
    model_ls: str = "{domain}/v2/models"
    status: str = "{domain}/v2/models/{model_nm}/status"
    infer: str = "{domain}/v2/models/{model_nm}/infer"
    explain: str = "{domain}/v2/models/{model_nm}/explain"


@dataclass
class RuntimeUrlFormat:
    live: str = "{domain}/ifservice/ready/{model_id}"
    infer_json: str = "{domain}/ifservice/predict2/{model_id}"
    infer_data: str = "{domain}/ifservice/infer/{model_id}"
    explain: str = "{domain}/ifservice/explain/{model_id}"




class KFServingClient:
    def __init__(self, domain: str):
        self.domain = domain
        self.url_format: KFServingUrlFormat = KFServingUrlFormat()
        # _url_format = {
        #     "live": "{domain}/health/live",
        #     "model_ls": "{domain}/v2/models",
        #     "status": "{domain}/v2/models/{model_nm}/status",
        #     "infer": "{domain}/v2/models/{model_nm}/infer",
        #     "explain": "{domain}/v2/models/{model_nm}/explain",
        # }
        # self.url_format = namedtuple(
        #     "UrlFormat",
        #     _url_format.keys())(*_url_format.values())

        self.host_format = "{model_nm}.{namespace}.example.com"

    def infer(self,
            model_nm: str,
            data: Union[Dict, np.ndarray, bytes],
            namespace: str = "default",
            ) -> requests.Response:
        headers = {
            "Host": self.host_format.format(
                model_nm=model_nm,
                namespace=namespace,
            )
        }
        url = self.url_format.infer.format(
            domain=self.domain,
            model_nm=model_nm,
        )
        if data is None:
            raise ValueError(
                "keyword argument 'data' must be given.")
        else:
            if isinstance(data, Dict):
                headers["Content-Type"] = "application/json"
                r = requests.post(
                    url,
                    data=json.dumps(data),
                    headers=headers,
                )
            elif isinstance(data, np.ndarray):
                with TemporaryFile() as npz_byte:
                    np.savez_compressed(npz_byte, instances=data)
                    # Only needed here to simulate closing & reopening file
                    npz_byte.seek(0)
                    m = MultipartEncoder(
                        fields={
                            "instances": ("instances.npz", npz_byte),
                        }
                    )
                    multipart_headers = {
                        "ce-specversion": "1.0",
                        "ce-source": "none",
                        "ce-type": "none",
                        "ce-id": "none",
                        "Content-Type": m.content_type,
                    }
                    headers = dict(headers, **multipart_headers)
                    r = requests.post(
                        url,
                        data=m,
                        headers=headers,
                    )
            else:
                raise TypeError(
                    "'data' must be one of {dict, np.ndarray, bytes}.")

            return r


    async def infer_async(self,
              model_nm: str,
              data: Union[Dict, np.ndarray, bytes],
              namespace: str = "default",
              session: aiohttp.ClientSession=None,
              ) -> Coroutine:
        headers = {
            "Host": self.host_format.format(
                model_nm=model_nm,
                namespace=namespace,
            )
        }
        url = self.url_format.infer.format(
            domain=self.domain,
            model_nm=model_nm,
        )
        if data is None:
            raise ValueError(
                "keyword argument 'data' must be given.")
        else:
            if isinstance(data, Dict):
                headers["Content-Type"] = "application/json"
                async with session.post(
                        url,
                        data=json.dumps(data),
                        headers=headers,
                        ) as resp:
                    r_text = await resp.text()
            elif isinstance(data, np.ndarray):
                with TemporaryFile() as npz_byte:
                    np.savez_compressed(npz_byte, instances=data)
                    # Only needed here to simulate closing & reopening file
                    npz_byte.seek(0)
                    m = MultipartEncoder(
                        fields={
                            "instances": ("instances.npz", npz_byte),
                        }
                    )
                    multipart_headers = {
                        "ce-specversion": "1.0",
                        "ce-source": "none",
                        "ce-type": "none",
                        "ce-id": "none",
                        "Content-Type": m.content_type,
                    }
                    headers = dict(headers, **multipart_headers)
                    # with aiohttp.MultipartWriter('form-data') as m:
                    #     # m.append(npz_byte, headers={
                    #     #          "Content-Type": "multipart/form-data"})
                    #     m.append_form(
                    #         [("instances.npz", npz_byte)],
                    #         headers={"Content-Type": "multipart/form-data"},
                    #     )
                    #     m.set_content_disposition('form-data', name="instances")
                    m = aiohttp.FormData()
                    m.add_field('instances',
                                   npz_byte,
                                   filename='instances.npz',
                                   content_type='multipart/form-data')
                    async with session.post(
                            url,
                            data=m,
                            headers=headers,
                            ) as resp:
                        r_text = await resp.text()
            else:
                raise TypeError(
                    "'data' must be one of {dict, np.ndarray, bytes}.")

            return r_text

    def _save_np_as_filestream(self, ndarray: np.ndarray):
        # npz_as_bytes = TemporaryFile()
        with TemporaryFile() as npz_as_bytes:
            np.savez_compressed(npz_as_bytes, instances=ndarray)
            # Only needed here to simulate closing & reopening file
            npz_as_bytes.seek(0)
            return npz_as_bytes


class RuntimeClient:
    def __init__(self, domain: str):
        self.domain = domain
        self.url_format: RuntimeUrlFormat = RuntimeUrlFormat()
        # _url_format = {
        #     "live": "{domain}/ifservice/ready/{model_id}",
        #     "infer_json": "{domain}/ifservice/predict2/{model_id}",
        #     "infer_data": "{domain}/ifservice/infer/{model_id}",
        #     "explain": "{domain}/ifservice/explain/{model_id}",
        # }
        # self.url_format = namedtuple(
        #     "UrlFormat",
        #     _url_format.keys())(*_url_format.values())
        self.auth_format = "Bearer {token}"

    def infer(self,
            model_id: str,
            data: Union[Dict, np.ndarray, bytes],
            token: str,
            ) -> requests.Response:
        headers = {
            "Authorization": self.auth_format.format(
                token=token,
            )
        }
        if data is None:
            raise ValueError(
                "keyword argument 'data' must be given.")
        else:
            if isinstance(data, Dict):
                headers["Content-Type"] = "application/json"
                url = self.url_format.infer_json.format(
                    domain=self.domain,
                    model_id=model_id,
                )
                r = requests.post(
                    url,
                    data=json.dumps(data),
                    headers=headers,
                )
            elif isinstance(data, np.ndarray):
                with TemporaryFile() as npz_byte:
                    np.savez_compressed(npz_byte, instances=data)
                    # Only needed here to simulate closing & reopening file
                    npz_byte.seek(0)
                    m = MultipartEncoder(
                        fields={
                            "instances": ("instances.npz", npz_byte),
                        }
                    )

                    headers["Content-Type"] = m.content_type
                    url = self.url_format.infer_data.format(
                        domain=self.domain,
                        model_id=model_id,
                    )
                    r = requests.post(
                        url,
                        data=m,
                        headers=headers,
                    )
            else:
                raise TypeError(
                    "'data' must be one of {dict, np.ndarray, bytes}.")

            return r

    async def infer_async(self,
              model_id: str,
              data: Union[Dict, np.ndarray, bytes],
              token: str,
              session: aiohttp.ClientSession=None,
              ) -> Coroutine:
        headers = {
            "Authorization": self.auth_format.format(
                token=token,
            )
        }
        if data is None:
            raise ValueError(
                "keyword argument 'data' must be given.")
        else:
            if isinstance(data, Dict):
                headers["Content-Type"] = "application/json"
                url = self.url_format.infer_json.format(
                    domain=self.domain,
                    model_id=model_id,
                )
                async with session.post(
                        url,
                        data=json.dumps(data),
                        headers=headers,
                        ) as resp:
                    r_text = await resp.text()
            elif isinstance(data, np.ndarray):
                with TemporaryFile() as npz_byte:
                    np.savez_compressed(npz_byte, instances=data)
                    # Only needed here to simulate closing & reopening file
                    npz_byte.seek(0)
                    m = MultipartEncoder(
                        fields={
                            "instances": ("instances.npz", npz_byte),
                        }
                    )

                    headers["Content-Type"] = m.content_type
                    url = self.url_format.infer_data.format(
                        domain=self.domain,
                        model_id=model_id,
                    )
                    # with aiohttp.MultipartWriter('form-data') as mpwriter:
                    #     mpwriter.append_form([('key', 'value')])
                    m = aiohttp.FormData()
                    m.add_field('instances',
                                npz_byte,
                                filename='instances.npz',
                                content_type='multipart/form-data')
                    async with session.post(
                            url,
                            data=m,
                            headers=headers,
                            ) as resp:
                        r_text = await resp.text()
            else:
                raise TypeError(
                    "'data' must be one of {dict, np.ndarray, bytes}.")

            return r_text

async def get_pokemon(session, url):
    async with session.get(url) as resp:
        pokemon = await resp.json()
        return pokemon['name']
