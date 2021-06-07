from runtime_client import KFServingClient, RuntimeClient
import numpy as np

data = np.ones([1, 224, 224, 3])

aa = KFServingClient("http://localhost:28080")
nn = np.load('input.npz')
bb = nn[nn.files[0]]
rr = aa.infer('test', data=bb)
