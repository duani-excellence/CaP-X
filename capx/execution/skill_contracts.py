"""Parameterized skill obligations on the existing dependency graph.

No environment, task name, motion recipe or score is available here. Physical
effects are established only by the trusted verifier, never by a returned call.
Contracts declare capabilities; the parent adapter still checks each primitive.
"""
import copy
import uuid
import time
from .dependency_graph import DependencyGraph


class SkillBoundaryStop(RuntimeError):
    """Trusted skill stopped at a known boundary, with public verified state.

    Never construct this from actor output or an unclassified transport failure.
    Unknown in-flight effects continue to quarantine the episode.
    """
    def __init__(self,reason,effects):
        super().__init__(reason)
        self.effects=copy.deepcopy(effects)


def validate_contract_task(value):
    d=copy.deepcopy(value)
    if d.get('schema')!='capx.contract_task.v1':raise ValueError('Unsupported contract schema')
    if not d.get('objects') or not d.get('goals'):raise ValueError('Objects and goals required')
    rows=d.get('intents',[]);ids=[r['id'] for r in rows]
    if not rows or len(set(ids))!=len(ids):raise ValueError('Unique intent ids required')
    resources=set(d.get('resources',[]));by_id={r['id']:r for r in rows}
    for row in rows:
        if not row.get('skill') or not row.get('effects'):raise ValueError('Skill/effects required')
        if not set(row.get('resources',[]))<=resources:raise ValueError('Unknown resource')
        if not set(row.get('after',[]))<=set(ids):raise ValueError('Unknown predecessor')
        for effect,dependencies in row.get('effect_dependencies',{}).items():
            if effect not in row['effects'] or not set(dependencies)<=set(row.get('requires',[])):
                raise ValueError('Invalid live effect dependency')
    def visit(i,path):
        if i in path:raise ValueError('Dependency cycle')
        for p in by_id[i].get('after',[]):visit(p,path+[i])
    for i in ids:visit(i,[])
    if not set(d['goals'])<=set(k for r in rows for k in r['effects']):raise ValueError('Goal lacks skill effect')
    return d


class ContractGraph:
    def __init__(self,definition,graph=None):
        self.definition=validate_contract_task(definition)
        self.graph=graph or DependencyGraph()
        self.intents={};self.intent_nodes={};self.conditions={}
        self.completed={};self.permissions={};self.locks={};self.quarantined=False;self.stopped=set()
        self.repair_signatures={};self.repair_count=0
        self.program_node=None
        self.goal_node=self.graph.add('task_goal',{'goals':self.definition['goals']})
        pending=list(self.definition['intents'])
        while pending:
            row=next(r for r in pending if set(r.get('after',[]))<=set(self.intents))
            self.register(row);pending.remove(row)

    def register_program(self,code,source):
        self.program_node=self.graph.add('program',dict(code=code,source=source))
        return self.program_node

    def register(self,row):
        row=copy.deepcopy(row);key=row['id']
        if key in self.intents:raise ValueError('Intent ids are immutable')
        if not set(row.get('resources',[]))<=set(self.definition['resources']):raise ValueError('Unknown resource')
        if not set(row.get('after',[]))<=set(self.intents):raise ValueError('Unknown predecessor')
        self.intents[key]=row
        parents=[self.goal_node]+[(self.intent_nodes[p],'precondition') for p in row.get('after',[])]
        self.intent_nodes[key]=self.graph.add('skill_intent',row,parents=parents)

    def observe(self,key,truth,evidence,parents=()):
        if truth not in ('TRUE','FALSE','UNKNOWN'):raise ValueError('Invalid truth')
        old=self.conditions.get(key)
        # Even TRUE->TRUE is a new permission generation; final capture must
        # revalidate a lease against that generation before physical dispatch.
        if old:self.graph.invalidate([old],reason='condition evidence replaced')
        node=self.graph.add('condition',dict(key=key,truth=truth,evidence=copy.deepcopy(evidence)),key=key,parents=parents)
        self.conditions[key]=node
        return node

    def truth(self,key):
        n=self.graph.nodes.get(self.conditions.get(key),{})
        return n.get('data',{}).get('truth','UNKNOWN') if n.get('valid') else 'UNKNOWN'

    def complete(self):return all(self.truth(k)=='TRUE' for k in self.definition['goals'])

    def order_satisfied(self,row):
        for predecessor in row.get('after',[]):
            if predecessor in self.completed:continue
            if row.get('allow_verified_predecessors',False) and all(self.truth(k)=='TRUE' for k in self.intents[predecessor]['effects']):continue
            return False
        return True

    def admit(self,intent):
        if self.quarantined:raise ValueError('Physical effect quarantined')
        if intent in self.completed:raise ValueError('Intent already completed; use new repair id')
        if intent in self.stopped:raise ValueError('Stopped intent requires new repair id')
        if any(p['intent']==intent for p in self.permissions.values()):raise ValueError('Intent already in flight')
        row=self.intents[intent]
        if not self.order_satisfied(row):raise ValueError('Unfinished predecessor')
        if any(r in self.locks for r in row.get('resources',[])):raise ValueError('Robot resource busy')
        keys=row.get('requires',[])
        if any(self.truth(k)!='TRUE' for k in keys):raise ValueError('Missing precondition evidence')
        token=uuid.uuid4().hex;deps={k:self.conditions[k] for k in keys}
        parents=list(deps.values())+[self.intent_nodes[intent]]
        if self.program_node:parents.append(self.program_node)
        node=self.graph.add('plan',dict(intent=intent,permission=token),parents=parents)
        self.permissions[token]=dict(intent=intent,dependencies=deps,node=node)
        for r in row.get('resources',[]):self.locks[r]=token
        return token

    def check_permission(self,token):
        p=self.permissions[token]
        if self.quarantined or not self.graph.nodes[p['node']]['valid'] or any(
            self.conditions.get(k)!=n or self.truth(k)!='TRUE' for k,n in p['dependencies'].items()):
            raise ValueError('stale physical permission')
        return p['intent']

    def _unlock(self,token):
        self.locks={r:t for r,t in self.locks.items() if t!=token}

    def finish(self,token,verified_effects):
        p=self.permissions.pop(token);row=self.intents[p['intent']]
        call=self.graph.add('call',dict(intent=p['intent'],returned=True),parents=[(p['node'],'attestation')],state='done')
        self.completed[p['intent']]=call;self._unlock(token)
        for key in row['effects']:
            parents=[self.conditions[k] for k in row.get('effect_dependencies',{}).get(key,[]) if k in self.conditions]
            parents.append((call,'attestation'))
            verified=verified_effects.get(key,'UNKNOWN')
            truth=verified.get('truth','UNKNOWN') if isinstance(verified,dict) else verified
            evidence=copy.deepcopy(verified.get('evidence',{})) if isinstance(verified,dict) else {}
            evidence.update(call=call,verifier='parent_verifier')
            self.observe(key,truth,evidence,parents=parents)

    def abort(self,token,known_effect):
        p=self.permissions.pop(token);self._unlock(token)
        self.stopped.add(p['intent'])
        call=self.graph.add('call',dict(intent=p['intent'],returned=False,known_effect=known_effect),parents=[(p['node'],'attestation')],state='done')
        if not known_effect:self.quarantined=True
        return call

    def recovery_scope(self,missing,group):
        pending=[r for i,r in self.intents.items() if i not in self.completed]
        all_keys=list(dict.fromkeys(k for r in pending for k in r.get('requires',[])))
        if group=='G0':return all_keys
        if group!='G1':raise ValueError('Unknown graph policy')
        return [k for k in all_keys if k in missing]

    def repair_obligation(self,max_repairs=8):
        """Regress broken aggregate goals, preserving true sibling effects.

        A new intent is a NEW physical obligation, never replay of an old call.
        Unknown physical effects remain quarantined. Identical truth frontiers
        cannot buy progress merely through new node/source IDs.
        """
        if self.quarantined or self.repair_count>=max_repairs:return None
        def regress(goal,path):
            if self.truth(goal)=='TRUE' or goal in path:return None
            producers=[r for r in self.definition['intents'] if goal in r['effects']]
            if len(producers)!=1:return None
            row=producers[0]
            for dependency in row.get('effect_dependencies',{}).get(goal,[]):
                if self.truth(dependency)!='TRUE':return regress(dependency,path+[goal])
            return goal,row
        for goal in self.definition['goals']:
            selected=regress(goal,[])
            if selected is None:continue
            key,row=selected
            signature=tuple((k,self.truth(k)) for k in [key]+row.get('requires',[]))
            if self.repair_signatures.get(key)==signature:continue
            self.repair_signatures[key]=signature;self.repair_count+=1
            repair=copy.deepcopy(row);repair['repair_of']=row['id']
            repair['id']='repair-'+uuid.uuid4().hex
            self.register(repair)
            return repair['id']
        return None

    def complexity(self):
        rows=self.intents;depth={}
        def level(i):
            if i not in depth:depth[i]=1+max([level(p) for p in rows[i].get('after',[])],default=0)
            return depth[i]
        for i in rows:level(i)
        children={i:sum(i in r.get('after',[]) for r in rows.values()) for i in rows}
        consumers={k:sum(k in r.get('requires',[]) for r in rows.values()) for r in rows.values() for k in r.get('requires',[])}
        return dict(skills=len(rows),depth=max(depth.values()),width=max(list(depth.values()).count(v) for v in depth.values()),
            forks=sum(n>1 for n in children.values()),joins=sum(len(r.get('after',[]))>1 for r in rows.values()),
            independent_goals=sum(not r.get('after') for r in rows.values()),
            terminal_goals=len(self.definition['goals']),condition_forks=sum(n>1 for n in consumers.values()),
            shared_conditions=sum(n>1 for n in consumers.values()))


class ContractBroker:
    """Parent-owned skill boundary; reuse with SandboxProcess's RPC callback.

    Capabilities and verifier are trusted implementations. The actor can select
    only an existing declared intent ID, never supply a truth value or resource
    lease. Multi-motion skills must themselves check their primitive boundaries.
    """
    def __init__(self,contracts,capabilities,verify,*,max_calls=64,seconds=600,emit=None,sync_invalidates=False):
        self.contracts=contracts;self.capabilities=capabilities;self.verify=verify
        self.max_calls=max_calls;self.calls=0;self.deadline=time.monotonic()+seconds
        self.emit=emit or (lambda row:None)
        self.sync_invalidates=sync_invalidates

    def refresh(self,keys):
        rows=self.verify(list(keys))
        for key in keys:
            row=rows.get(key,dict(truth='UNKNOWN',evidence={'reason':'missing_verifier'}))
            self.contracts.observe(key,row['truth'],row['evidence'])
        return rows

    def execute(self,intent_id):
        if self.calls>=self.max_calls or time.monotonic()>=self.deadline:raise ValueError('Skill budget exhausted')
        c=self.contracts;row=c.intents[intent_id]
        if row['skill'] not in self.capabilities:raise ValueError('Unqualified skill capability')
        # The last capture/verification happens BEFORE acquiring the live lease.
        self.refresh(row.get('requires',[]))
        token=c.admit(intent_id);c.check_permission(token)
        started=time.monotonic();self.calls+=1
        try:
            self.capabilities[row['skill']](copy.deepcopy(row))
            for key in row.get('invalidates',[]):
                c.observe(key,'UNKNOWN',dict(reason='declared_skill_effect',intent=intent_id))
            keys=list(dict.fromkeys(row['effects']+(row.get('invalidates',[]) if self.sync_invalidates else [])))
            effects=self.verify(keys)
            c.finish(token,effects)
            if self.sync_invalidates:
                call=c.completed[intent_id]
                for key in row.get('invalidates',[]):
                    v=effects.get(key,dict(truth='UNKNOWN',evidence={'reason':'missing_verifier'}))
                    c.observe(key,v['truth'],v['evidence'],parents=[(call,'attestation')])
            self.emit(dict(intent=intent_id,status='returned',effects=effects,seconds=time.monotonic()-started))
            return dict(effects={k:c.truth(k) for k in row['effects']},call_id=c.completed[intent_id])
        except SkillBoundaryStop as exc:
            call=c.abort(token,known_effect=True) if token in c.permissions else None
            for key in row.get('invalidates',[]):
                c.observe(key,'UNKNOWN',dict(reason='possible_partial_skill_effect',intent=intent_id))
            for key,value in exc.effects.items():
                c.observe(key,value['truth'],value['evidence'],parents=[(call,'attestation')] if call else [])
            self.emit(dict(intent=intent_id,status='known_boundary_stop',reason=str(exc),
                           effects=exc.effects,seconds=time.monotonic()-started))
            raise
        except BaseException:
            if token in c.permissions:c.abort(token,known_effect=False)
            self.emit(dict(intent=intent_id,status='unknown_effect_quarantined',seconds=time.monotonic()-started))
            raise
