"""Opt-in read-only graph telemetry. Never supplies evidence or action permission."""
import contextvars
import json
import logging
import time
from pathlib import Path

_ACTIVE = contextvars.ContextVar('capx_graph_journal', default=None)

class GraphJournal:
    def __init__(self,path):
        self.path=Path(path);self.graphs=[];self.count=0;self.error=None
    def __enter__(self):
        self.stream=None;self.token=_ACTIVE.set(self);return self
    def __exit__(self,*args):
        _ACTIVE.reset(self.token)
        if self.stream:self.stream.close()
    def record(self,graph,event):
        if self.error:return
        try:
            if self.stream is None:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                self.stream=self.path.open('x')
            index=next((i for i,g in enumerate(self.graphs) if g is graph),None)
            if index is None:index=len(self.graphs);self.graphs.append(graph)
            self.count+=1
            self.stream.write(json.dumps(dict(index=self.count,unix=time.time(),graph_id=f'g{index}',event=event),ensure_ascii=False)+'\n')
            self.stream.flush()
        except (OSError,ValueError,TypeError) as exc:
            self.error=type(exc).__name__
            logging.getLogger(__name__).exception('Graph telemetry failed; does not change control semantics')

def publish(graph,event):
    journal=_ACTIVE.get()
    if journal is not None:journal.record(graph,event)
