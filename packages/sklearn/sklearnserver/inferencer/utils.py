import numpy as np
import pandas as pd

def change_ndarray_tolist(sub):
    if isinstance(sub, list):
        return [change_ndarray_tolist(x) for x in sub]
    elif isinstance(sub, tuple):
        return tuple(change_ndarray_tolist(x) for x in sub)
    elif isinstance(sub, set):
        return {change_ndarray_tolist(x) for x in sub}
    elif isinstance(sub, dict):
        return {k: change_ndarray_tolist(v) for k, v in sub.items()}
    elif isinstance(sub, [pd.DataFrame, pd.Series, np.ndarray]):
        return sub.tolist()
    else:
        return sub
