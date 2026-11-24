"""Exact-input evidence reuse and attribution, without inferred physical truth.

Semantic masks are reused only for identical current RGB and calibration.
Fresh depth/robot state still undergo the existing geometric safety checks.
"""
import contextvars
import copy
import functools
import hashlib
from collections import OrderedDict

import numpy as np

_CONTEXT=contextvars.ContextVar('perception_evidence_context',default=())


def current_context():
    return list(_CONTEXT.get())


def evidence_scope(function):
    @functools.wraps(function)
    def wrapped(self,*args,**kwargs):
        row={'component':function.__qualname__}
        if args and isinstance(args[0],str):row['subject']=args[0]
        elif args and isinstance(args[0],list) and all(isinstance(x,str) for x in args[0]):row['conditions']=args[0]
        elif args and isinstance(args[0],dict):row['intent']={k:args[0][k] for k in ('id','skill','object','target') if k in args[0]}
        token=_CONTEXT.set(_CONTEXT.get()+(row,))
        try:return function(self,*args,**kwargs)
        finally:_CONTEXT.reset(token)
    return wrapped


class ExactEvidenceCache:
    def __init__(self,max_entries=256,max_bytes=128*1024*1024):
        self.entries=OrderedDict();self.max_entries=max_entries;self.max_bytes=max_bytes;self.bytes=0

    @staticmethod
    def digest(value):
        h=hashlib.sha256()
        def add(v):
            if isinstance(v,np.ndarray):
                h.update(str((v.dtype.str,v.shape)).encode());h.update(np.ascontiguousarray(v).tobytes())
            elif isinstance(v,dict):
                for k in sorted(v):h.update(str(k).encode());add(v[k])
            elif isinstance(v,(list,tuple)):
                for x in v:add(x)
            else:h.update(repr(v).encode())
            h.update(b'\0')
        add(value);return h.hexdigest()

    @staticmethod
    def size(value):
        if isinstance(value,np.ndarray):return value.nbytes
        if isinstance(value,dict):return sum(ExactEvidenceCache.size(v) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(ExactEvidenceCache.size(v) for v in value)
        return 64

    def get_or_compute(self,key,compute,source):
        if key in self.entries:
            value,origin,size=self.entries.pop(key);self.entries[key]=(value,origin,size)
            return copy.deepcopy(value),True,origin
        value=compute();size=self.size(value)
        if size<=self.max_bytes:
            while self.entries and (len(self.entries)>=self.max_entries or self.bytes+size>self.max_bytes):
                _,(_,_,old)=self.entries.popitem(last=False);self.bytes-=old
            self.entries[key]=(copy.deepcopy(value),source,size);self.bytes+=size
        return value,False,source
