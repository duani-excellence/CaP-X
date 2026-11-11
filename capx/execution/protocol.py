"""JSON only, never pickle. Shared by the isolated actor and trusted broker."""
import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        return {'array': value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {'tuple': [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {'array'}:
            arr = np.asarray(value['array'])
            if arr.size > 10000 or arr.dtype.kind not in 'biuf':
                raise ValueError('Invalid RPC array')
            return arr
        if set(value) == {'tuple'}:
            return tuple(decode(x) for x in value['tuple'])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(x) for x in value]
    return value
