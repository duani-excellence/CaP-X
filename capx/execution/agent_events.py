"""Durable append-only execution journal. Checkpoints never restore physics."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time


def plain(value):
    return json.loads(json.dumps(value,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x),allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class EventStore:
    def __init__(self,root,*,resume=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'events.jsonl';self.sequence=0;self.previous='0'*64;self.actions={}
        if self.path.exists() and not resume:raise FileExistsError(self.path)
        if resume and self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    row=json.loads(line);signature=row.pop('sha256')
                    if row['sequence']!=self.sequence+1 or row['previous']!=self.previous or digest(row)!=signature:raise ValueError()
                except (ValueError,KeyError):raise ValueError('Event integrity failure; manual reconciliation required') from None
                self.sequence=row['sequence'];self.previous=signature;self._apply(row)
            # A lost worker cannot certify whether its last dispatched motion ran.
            for key,row in list(self.actions.items()):
                if row['state']=='IN_FLIGHT':self.finish_action(key,'UNKNOWN',{'reason':'worker_restart_requires_current_state_reconciliation'})

    def _apply(self,row):
        d=row['data']
        if row['type'] in ('ActionStarted','ActionResult'):
            self.actions.setdefault(d['action_id'],{}).update(d)

    def append(self,kind,data):
        row=dict(sequence=self.sequence+1,type=kind,unix=time.time(),previous=self.previous,data=plain(data))
        signature=digest(row)
        with self.path.open('a') as f:
            f.write(json.dumps(dict(row,sha256=signature),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        self.sequence+=1;self.previous=signature;self._apply(row)
        return row['sequence']

    def unresolved(self):
        return [k for k,v in self.actions.items() if v.get('physical') and v['state'] in ('IN_FLIGHT','UNKNOWN')]

    def start_action(self,key,name,inputs,*,physical=False,site=None):
        if key in self.actions:raise ValueError('Action ID already used; physical calls cannot be replayed')
        if physical and self.unresolved():raise ValueError('Unknown physical effect requires reconciliation')
        data=dict(action_id=key,name=name,inputs=inputs,physical=physical,site=site,state='PENDING')
        self.append('ActionStarted',data);self.append('ActionStarted',dict(data,state='IN_FLIGHT'))

    def finish_action(self,key,state,result):
        if state not in ('DONE','FAILED','UNKNOWN'):raise ValueError('Invalid action lifecycle')
        if key not in self.actions:raise ValueError('No dispatch record')
        self.append('ActionResult',dict(action_id=key,state=state,result=result))

    def checkpoint(self,state):
        p=self.root/f'checkpoint-{self.sequence+1:06d}.json'
        with p.open('x') as f:
            json.dump(dict(schema='capx.harness.checkpoint.v33',event_sequence=self.sequence,event_hash=self.previous,
                physical_rollback=False,state=plain(state),actions=self.actions),f,ensure_ascii=False,indent=2)
            f.flush();os.fsync(f.fileno())
        self.append('CheckpointSaved',{'path':p.name,'physical_rollback':False})
        tmp=self.root/'latest.json.tmp';tmp.write_text(json.dumps({'checkpoint':p.name,'event_sequence':self.sequence}))
        tmp.replace(self.root/'latest.json');return p

    def program(self,version,code,origin,request=None):
        p=self.root/f'program-{version:03d}.json'
        with p.open('x') as f:json.dump(dict(version=version,code=code,origin=origin,request=request,sha256=digest(code)),f,indent=2)
        self.append('PlanGenerated',dict(version=version,path=p.name,origin=origin,request=request,code_hash=digest(code)))
        return p
