"""Causal compilation and bounded local repair on ContractGraph.

Inspired by LOGIV's live support / regression / validate-before-splice design.
This module does not execute physics, acquire observations, or establish truth.
UNKNOWN is never closed-world false. Conditional validation records future
public-verification obligations; it is NOT an execution permission.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import heapq
import itertools
import json
import uuid


def program_order(code, known):
    """Bounded literal skill sequences, parsed without executing actor code.

    Accept direct calls or one-level for loops over immutable literal lists.
    No branching, dynamic iteration, mutation, reassignment or partial extraction.
    """
    order=[];constants={};reserved={'execute_skill','execution_state','refresh_evidence','repair_skill','print'}
    def literal_list(node):
        if not isinstance(node,(ast.List,ast.Tuple)) or len(node.elts)>len(known):
            raise ValueError('Plan sequence must be a bounded literal list')
        if not all(isinstance(x,ast.Constant) and isinstance(x.value,str) for x in node.elts):
            raise ValueError('Plan sequence must contain literal intent ids')
        return [x.value for x in node.elts]
    def add(key):
        if not isinstance(key,str) or key not in known or key in order:raise ValueError('Unknown or repeated initial intent')
        order.append(key)
    for node in ast.parse(code).body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant):continue
        if isinstance(node,ast.Assign):
            if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):raise ValueError('Unsupported plan assignment')
            name=node.targets[0].id
            if name in constants or name in reserved:raise ValueError('Plan variables cannot be reassigned')
            constants[name]=literal_list(node.value);continue
        if isinstance(node,ast.For):
            if not isinstance(node.target,ast.Name) or node.target.id in reserved or node.target.id in constants or node.orelse or len(node.body)!=1:
                raise ValueError('Unsupported plan loop')
            values=constants.get(node.iter.id) if isinstance(node.iter,ast.Name) else literal_list(node.iter)
            body=node.body[0]
            if values is None or not isinstance(body,ast.Expr) or not isinstance(body.value,ast.Call):raise ValueError('Unsupported plan loop body')
            call=body.value
            if not isinstance(call.func,ast.Name) or call.func.id!='execute_skill' or call.keywords or len(call.args)!=1 or not isinstance(call.args[0],ast.Name) or call.args[0].id!=node.target.id:
                raise ValueError('Loop must only dispatch its literal intent sequence')
            for key in values:add(key)
            continue
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):
            raise ValueError('Initial plan requires unconditional literal execute_skill calls')
        call=node.value
        if not isinstance(call.func,ast.Name):raise ValueError('Unsupported plan call')
        if call.func.id=='print' and not call.keywords and all(isinstance(x,ast.Constant) for x in call.args):continue
        if call.func.id!='execute_skill' or len(call.args)!=1 or call.keywords or not isinstance(call.args[0],ast.Constant):
            raise ValueError('Initial plan requires unconditional literal execute_skill calls')
        add(call.args[0].value)
    if not order:raise ValueError('Empty skill plan')
    return order


def required(row):
    result={k:'TRUE' for k in row.get('requires',[])+row.get('data_requires',[])}
    for k in row.get('requires_false',[]):
        if k in result:raise ValueError('Contradictory contract preconditions')
        result[k]='FALSE'
    return result


def changes(row):
    add=set(row.get('effects',[]));delete=set(row.get('deletes',[]))
    if add&delete:raise ValueError('Contradictory predicted effects')
    return dict({k:'FALSE' for k in delete},**{k:'TRUE' for k in add})


def apply_prediction(state,row):
    out=dict(state)
    for k in row.get('invalidates',[]):out[k]='UNKNOWN'
    out.update(changes(row))
    return out


def signature(row):
    # Stable across repair ids, attempts and source versions; only capability and
    # bound parameters/semantics matter for progress and exclusions.
    ignore={'id','after','repair_of','allow_verified_predecessors'}
    return hashlib.sha256(json.dumps({k:v for k,v in row.items() if k not in ignore},sort_keys=True).encode()).hexdigest()


def ground_library(definition):
    """Finite trusted capability bindings, independent of the initial program.

    The actor cannot submit or alter contracts. Bindings are object/resource
    domains and parameter compatibility, never a task solution or after chain.
    """
    result=[]
    def bind(value,values):
        if isinstance(value,str):return value.format_map(values)
        if isinstance(value,list):return [bind(v,values) for v in value]
        if isinstance(value,dict):return {bind(k,values):bind(v,values) for k,v in value.items()}
        return value
    for schema in definition.get('skill_library',[]):
        for values in schema.get('bindings',[{}]):
            row=bind(schema['contract'],values)
            required(row);changes(row)
            if row.get('after'):raise ValueError('Capability library cannot prescribe task flow')
            if not set(row.get('resources',[]))<=set(definition['resources']):raise ValueError('Unknown library resource')
            row['id']='cap-'+schema['name']+'-'+signature(row)[:12]
            result.append(row)
    return result


def validate_sequence(state,rows,goals,specs,*,conditional=False,completed=()):
    current=dict(state);obligations=[];seen=set(completed)
    for index,row in enumerate(rows):
        for predecessor in row.get('after',[]):
            if predecessor not in seen:return dict(valid=False,index=index,reason='order',condition=predecessor,obligations=obligations)
        for key,truth in required(row).items():
            actual=current.get(key,'UNKNOWN')
            if actual==truth:continue
            if conditional and actual=='UNKNOWN' and specs.get(key,{}).get('observable'):
                obligations.append(key);current[key]=truth
            else:return dict(valid=False,index=index,condition=key,expected=truth,actual=actual,reason='precondition',obligations=obligations)
        current=apply_prediction(current,row);seen.add(row['id'])
    for key in goals:
        if current.get(key,'UNKNOWN')=='TRUE':continue
        if conditional and current.get(key,'UNKNOWN')=='UNKNOWN' and specs.get(key,{}).get('observable'):
            obligations.append(key);continue
        return dict(valid=False,index=len(rows),condition=key,reason='goal',obligations=obligations)
    return dict(valid=True,obligations=list(dict.fromkeys(obligations)),predicted=current)


def regress(goals,rows):
    wanted=dict(goals)
    for row in reversed(rows):
        effect=changes(row)
        if any(k in effect and effect[k]!=v for k,v in wanted.items()):return None
        wanted={k:v for k,v in wanted.items() if effect.get(k)!=v}
        for key,value in required(row).items():
            if key in wanted and wanted[key]!=value:return None
            wanted[key]=value
    return wanted


class CausalPlan:
    """Versioned future plan indexed in the existing historical DependencyGraph."""
    def __init__(self,contracts,order,*,automatic=True,previous=None):
        self.c=contracts;self.order=list(order);self.automatic=automatic
        if len(set(order))!=len(order) or any(i not in contracts.intents for i in order):raise ValueError('Invalid plan ids')
        self.edges=[];self.revision='plan-'+uuid.uuid4().hex;self.nodes={}
        producer={};state={k:contracts.truth(k) for k in contracts.conditions}
        def edge(source,target,kind,condition=None):
            value=dict(source=source,target=target,kind=kind,condition=condition)
            if value not in self.edges:self.edges.append(value)
        for index,key in enumerate(order):
            row=contracts.intents[key]
            for condition,wanted in required(row).items():
                origin=producer.get(condition,'START')
                if automatic:
                    edge(origin,key,'data' if condition in row.get('data_requires',[]) else ('causal' if state.get(condition,'UNKNOWN')==wanted else 'evidence'),condition)
            for predecessor in row.get('after',[]):
                if predecessor in order:edge(predecessor,key,'order')
            for prior in order[:index]:
                other=contracts.intents[prior]
                if set(other.get('resources',[]))&set(row.get('resources',[])):edge(prior,key,'resource')
                writes=changes(other);other_reads=required(other);now=changes(row)
                if any(k in writes and writes[k]!=v for k,v in (required(row)|now).items()) or any(k in now and now[k]!=v for k,v in other_reads.items()):
                    edge(prior,key,'conflict')
            state=apply_prediction(state,row)
            for k in set(changes(row))|set(row.get('invalidates',[])):producer[k]=key
        if automatic:
            for k in contracts.definition['goals']:edge(producer.get(k,'START'),'GOAL','causal',k)
        parents=[(contracts.goal_node,'attestation')]
        if contracts.program_node:parents.append((contracts.program_node,'attestation'))
        self.node=contracts.graph.add('causal_plan',dict(revision=self.revision,order=self.order,edges=self.edges,
            automatic=automatic,previous=previous,predicted_only=True),parents=parents)
        for key in order:
            links=[(self.nodes[e['source']],'precondition') for e in self.edges if e['target']==key and e['source'] in self.nodes]
            sources=[(contracts.conditions[e['condition']],'attestation') for e in self.edges if e['target']==key and e['source']=='START' and e['condition'] in contracts.conditions]
            self.nodes[key]=contracts.graph.add('causal_action',dict(intent=key,predicted_effects=changes(contracts.intents[key]),
                edges=[e for e in self.edges if e['target']==key]),parents=[(self.node,'attestation')]+links+sources)

        contracts.future_plan=self

    def observe(self,key,node):
        for edge in self.edges:
            if edge['condition']!=key or edge['target'] in self.c.completed:continue
            parents=[node,(self.nodes.get(edge['target'],self.node),'attestation')]
            self.c.graph.add('live_causal_support',dict(revision=self.revision,condition=key,node=node,
                target=edge['target'],truth=self.c.truth(key)),parents=parents)

    def returned(self,intent,call,status):
        if intent in self.nodes:
            self.c.graph.add('causal_execution',dict(revision=self.revision,intent=intent,call=call,status=status),
                parents=[(self.nodes[intent],'attestation'),(call,'attestation')],state='done')

    def pending(self):
        return [i for i in self.order if i not in self.c.completed and i not in self.c.stopped]

    def live_support(self):
        done=set(self.c.completed)|{'START'}
        return [dict(e,expected=required(self.c.intents[e['target']]).get(e['condition'],'TRUE') if e['target'] in self.c.intents else 'TRUE')
                for e in self.edges if e['kind'] in ('causal','data','evidence') and e['source'] in done and e['target'] not in done]

    def affected(self,keys):
        pending=set(self.pending());reached={e['target'] for e in self.edges if e['condition'] in keys and e['target'] in pending}
        if not self.automatic:return list(self.pending())
        while True:
            old=set(reached)
            reached.update(e['target'] for e in self.edges if e['kind'] in ('causal','data','evidence') and e['source'] in reached and e['target'] in pending)
            if reached==old:break
        return [i for i in self.order if i in reached]

    def commit(self,decision):
        c=self.c
        if decision['status']!='ready':raise ValueError('Unvalidated patch')
        if c.quarantined or c.locks or c.permissions:raise ValueError('Unreconciled in-flight action')
        if decision['versions']!=c.conditions:raise ValueError('Plan state changed before patch commit')
        rows=copy.deepcopy(decision['fragment']);new_ids=[]
        for row in rows:
            row['id']='repair-'+uuid.uuid4().hex;row['repair_of']='capability:'+signature(row)
            # Library actions have no history predecessors. Existing retained
            # explicit constraints are unchanged and revalidated below.
            new_ids.append(row['id'])
        combined=rows+[c.intents[i] for i in decision['retained']]
        if decision.get('check_continuation',True):
            checked=validate_sequence({k:c.truth(k) for k in c.conditions},combined,c.definition['goals'],
                c.definition.get('condition_specs',{}),conditional=True,completed=c.completed)
            if not checked['valid']:raise ValueError('Incompatible patch at commit')
        for row in rows:
            required(row);changes(row)
            if not row.get('skill') or not row.get('effects'):raise ValueError('Invalid patch capability')
            if not set(row.get('resources',[]))<=set(c.definition['resources']):raise ValueError('Unknown patch resource')
            if row.get('after'):raise ValueError('Library patch must not contain history pointers')
        for row in rows:c.register(row)
        updated=CausalPlan(c,new_ids+decision['retained'],automatic=self.automatic,previous=self.revision)
        c.graph.add('causal_patch',dict(previous=self.revision,current=updated.revision,new_ids=new_ids,
            affected=decision['affected'],replaced=decision['replaced'],retained=decision['retained'],
            validation=decision['validation'],goal_regression=decision['regressed'],
            versions=decision['versions']),parents=[(self.node,'attestation'),(updated.node,'attestation')])
        return updated


def repair_plan(plan,library,*,failed=(),exhausted=None,forbidden=(),max_expansions=512,max_depth=6,check_continuation=True):
    """Regress the retained suffix, search finite capabilities, validate then splice.

    Effects of completed calls are never replayed. A stopped action is replaced;
    a causal descendant is merely affected. Bounded scope expansion removes only
    an incompatible affected suffix, preserving independent live branches.
    """
    c=plan.c;exhausted=exhausted or {};state={k:c.truth(k) for k in c.conditions};specs=c.definition.get('condition_specs',{})
    result=dict(status='blocked',reason='no_compatible_fragment',affected=plan.affected(failed),
        replaced=[i for i in plan.order if i in c.stopped],versions=dict(c.conditions),check_continuation=check_continuation)
    if c.quarantined or c.permissions or c.locks:return dict(result,reason='quarantined')
    result['failure_classes']={k:('tool_data_invalid' if specs.get(k,{}).get('kind')=='tool_data' else
        'physical_condition_violated' if c.truth(k)!='UNKNOWN' else 'evidence_missing_or_expired') for k in failed}
    if result['replaced']:result['failure_classes']['attempted_calls']='known_boundary_stop_new_id_required'
    retained=plan.pending();expansions=0
    for scope_round in range(3):
        keep=[c.intents[i] for i in retained]
        goal=regress({k:'TRUE' for k in c.definition['goals']},keep)
        if goal is None:
            # Threats in a retained suffix cannot be repaired by a prefix.
            return dict(result,reason='inconsistent_retained_suffix',retained=retained)
        immediate=set(required(keep[0])) if keep else set()
        deferred=[k for k in goal if k not in immediate and k not in failed and
                  state.get(k,'UNKNOWN')=='UNKNOWN' and specs.get(k,{}).get('observable') and specs[k].get('kind')=='tool_data']
        search_state=dict(state)
        for key in deferred:search_state[key]=goal[key]
        evidence=[k for k,v in goal.items() if k not in deferred and state.get(k,'UNKNOWN')=='UNKNOWN' and specs.get(k,{}).get('observable') and exhausted.get(k)!='UNKNOWN']
        if evidence:return dict(result,status='evidence',keys=evidence,retained=retained,regressed=goal)
        search_goal=goal if check_continuation else {k:v for k,v in goal.items() if state.get(k,'UNKNOWN')!=v}
        frontier=[(0.,0,0,search_state,[])];serial=itertools.count(1);seen=set();validation=None
        while frontier and expansions<max_expansions:
            cost,depth,_,now,fragment=heapq.heappop(frontier)
            fingerprint=tuple(sorted(now.items()))
            if fingerprint in seen:continue
            seen.add(fingerprint);expansions+=1
            if all(now.get(k,'UNKNOWN')==v for k,v in search_goal.items()):
                validation=validate_sequence(state,fragment+keep,c.definition['goals'],specs,conditional=True,completed=c.completed)
                if validation['valid'] or not check_continuation:
                    return dict(result,status='ready',reason='validated_local_fragment',fragment=fragment,retained=retained,
                        regressed=goal,validation=validation,deferred_evidence=deferred,expansions=expansions,cost=cost,scope_round=scope_round)
            if depth>=max_depth:continue
            for row in library:
                if signature(row) in forbidden and not fragment:continue
                # A verified physical prefix can create a new observation
                # opportunity. Future tool evidence is an explicit obligation,
                # never an assumed physical predicate or dispatch permission.
                def supported(k,v):
                    return now.get(k,'UNKNOWN')==v or (bool(fragment) and now.get(k,'UNKNOWN')=='UNKNOWN'
                        and specs.get(k,{}).get('observable') and specs[k].get('kind')=='tool_data')
                if any(not supported(k,v) for k,v in required(row).items()):continue
                if row.get('after'):continue
                nxt=apply_prediction(now,row)
                if nxt==now:continue
                heapq.heappush(frontier,(cost+max(.01,float(row.get('cost',1.))),depth+1,next(serial),nxt,fragment+[row]))
        # Expand the minimum physically affected region, never an unrelated goal.
        candidates=[i for i in retained if i in result['affected']]
        if not candidates:break
        first=candidates[0];remove=set(plan.affected(c.intents[first]['effects']))|{first}
        result['replaced']+= [i for i in retained if i in remove]
        retained=[i for i in retained if i not in remove]
    # A newly grounded recovery capability may need evidence absent from the
    # original task intents. Regress its inputs as well; source-id churn does
    # not clear the same-value exhaustion guard and no UNKNOWN licenses motion.
    needed=set(goal or {});visited=set();candidates=[]
    for _ in range(max_depth):
        upcoming=set()
        for row in library:
            if signature(row) in forbidden or not needed.intersection(changes(row)):continue
            req=required(row);unknown=[k for k,v in req.items() if state.get(k,'UNKNOWN')=='UNKNOWN']
            if any(state.get(k,'UNKNOWN') not in ('UNKNOWN',v) for k,v in req.items()):
                upcoming.update(k for k,v in req.items() if state.get(k,'UNKNOWN')!=v);continue
            keys=[k for k in unknown if specs.get(k,{}).get('observable') and exhausted.get(k)!='UNKNOWN']
            if keys and len(keys)==len(unknown):candidates.append((float(row.get('cost',1.)),keys))
            upcoming.update(unknown)
        visited.update(needed);needed=upcoming-visited
        if candidates or not needed:break
    if candidates and expansions<max_expansions:
        _,keys=min(candidates,key=lambda x:(x[0],len(x[1])))
        return dict(result,status='evidence',keys=keys,retained=retained,regressed=goal,
                    reason='new_capability_prerequisite_evidence',expansions=expansions)
    return dict(result,retained=retained,expansions=expansions,reason='search_budget' if expansions>=max_expansions else result['reason'])
