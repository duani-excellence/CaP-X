"""Actual HTTP attempts (including retries), scoped to a worker episode."""
import contextvars
import json
import threading
import time
from urllib.parse import urlsplit, urlunsplit
import requests

_ACTIVE = contextvars.ContextVar('capx_request_audit', default=None)

class RequestAudit:
    def __init__(self, path):
        self.path = path
        self.count = self.llm_count = 0
        self.lock = threading.Lock()
    def __enter__(self):
        self.token = _ACTIVE.set(self)
        return self
    def __exit__(self, *args):
        _ACTIVE.reset(self.token)
    def record(self, url, payload, status, error, elapsed):
        # Never persist request headers, credentials, or URL query strings.
        parts = urlsplit(url)
        safe_url = urlunsplit((parts.scheme, parts.hostname + ((':'+str(parts.port)) if parts.port else ''), parts.path, '', ''))
        with self.lock:
            self.count += 1
            self.llm_count += int(isinstance(payload, dict) and 'model' in payload)
            row=dict(attempt=self.count,url=safe_url,payload=payload,status=status,error=error,elapsed_seconds=elapsed)
            with self.path.open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')


def audited_post(url, **kwargs):
    audit = _ACTIVE.get()
    start=time.monotonic(); response=None; error=None
    try:
        response=requests.post(url, **kwargs)
        return response
    except Exception as exc:
        error=type(exc).__name__
        raise
    finally:
        if audit is not None:
            payload=kwargs.get('json',kwargs.get('data'))
            if isinstance(payload,str):
                try: payload=json.loads(payload)
                except ValueError: pass
            audit.record(url,payload,getattr(response,'status_code',None),error,time.monotonic()-start)
