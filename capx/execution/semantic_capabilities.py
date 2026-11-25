"""Untrusted semantic proposals, bound to real public APIs by the parent.

Stored code only executes in the existing Bubblewrap Sandbox through the original
broker. Semantic edges are hypotheses, not runtime data or causal evidence. This
registry cannot write condition truth or grant motion permissions.
"""
import ast
import copy
import fcntl
import inspect
import json
from pathlib import Path
import re
import time
import uuid
from .agent_events import plain, digest
from .sandbox import SandboxProcess, validate


TYPES={'string':lambda x:isinstance(x,str),'number':lambda x:type(x) in (float,int),
       'integer':lambda x:type(x) is int,'boolean':lambda x:type(x) is bool,
       'array':lambda x:isinstance(x,list),'object':lambda x:isinstance(x,dict),
       'null':lambda x:x is None}
PURE={'range','len','min','max','sum','abs','round','float','int','str','bool','list','dict','enumerate','zip','sorted','print','ValueError'}


def tool_manifest(functions,effects):
    result={}
    for name,fn in functions.items():
        try:source=inspect.getsource(fn)
        except (OSError,TypeError):source=type(fn).__module__+'.'+type(fn).__qualname__
        try:signature=str(inspect.signature(fn))
        except (TypeError,ValueError):signature='unavailable'
        try:source+=Path(inspect.getsourcefile(fn)).read_text()
        except (OSError,TypeError):pass
        result[name]=dict(name=name,signature=signature,implementation_hash=digest(source),
            physical_side_effects=effects.get(name,['parent_guarded_possible_physical_effect']),
            doc=(inspect.getdoc(fn) or '')[:1600])
    return result


def check_values(values,schema):
    return isinstance(values,dict) and set(values)==set(schema) and all(TYPES[t](values[k]) for k,t in schema.items())


def output_path(field,schema):
    return [field] if field in schema else field.split('.')


def output_leaves(value,path=()):
    if isinstance(value,dict) and value:
        return [leaf for k,v in value.items() for leaf in output_leaves(v,path+(str(k),))]
    return [path]


def string_values(value):
    if isinstance(value,str):return {value}
    if isinstance(value,dict):return set().union(set(),*(string_values(v) for v in value.values()))
    if isinstance(value,(list,tuple)):return set().union(set(),*(string_values(v) for v in value))
    return set()


def result_ids(value):
    if isinstance(value,dict):
        return ({value['id']} if isinstance(value.get('id'),str) else set()) | set().union(set(),*(result_ids(v) for v in value.values()))
    if isinstance(value,list):return set().union(set(),*(result_ids(v) for v in value))
    return set()


def result_dependencies(value,graph):
    if not isinstance(value,dict):return []
    refs=value.get('dependency_refs',[])
    if not isinstance(refs,list):raise ValueError('Invalid dependency_refs protocol')
    refs=list(refs)
    if value.get('result_ref') is not None:refs.append(value['result_ref'])
    # Existing id-returning APIs are the documented legacy alias.
    if isinstance(value.get('id'),str) and value['id'] in graph.nodes:refs.append(value['id'])
    if any(not isinstance(r,str) or r not in graph.nodes for r in refs):raise ValueError('Unknown public result provenance')
    return list(dict.fromkeys(refs))


def semantic_case(value):
    """Exclude journal identity, retain actual measurements and model versions."""
    metadata={'id','result_ref','dependency_refs','provenance_refs','source','source_id','node',
              'timestamp','time','unix','path','directory','episode','output_node','record_id','candidate_id'}
    if isinstance(value,dict):return {k:semantic_case(v) for k,v in value.items() if k not in metadata}
    if isinstance(value,list):return [semantic_case(v) for v in value]
    return value


class CapabilityRegistry:
    def __init__(self,root,functions,*,graph,tool_effects,episode,callback=None,
                 evidence_valid=None,condition_state=None,emit=None,seconds=None,qualify=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.functions=dict(functions);self.graph=graph;self.episode=str(episode)
        self.manifest=tool_manifest(functions,tool_effects);self.callback=callback or (lambda n,a,k:self.functions[n](*a,**k))
        self.evidence_valid=evidence_valid or (lambda ref:ref in graph.nodes and graph.nodes[ref]['valid'])
        self.condition_state=condition_state or (lambda:{})
        self.emit=emit or (lambda *a,**k:None);self.seconds=seconds or (lambda:60.)
        self.nodes={};self.running=False;self.qualify=qualify

    def event(self,kind,**data):
        row=plain(dict(id=uuid.uuid4().hex,time=time.time(),episode=self.episode,kind=kind,**data))
        with (self.root/'events.jsonl').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        self.emit('SemanticCapabilityEvent',record=row)
        return row

    def register(self,proposal):
        errors=[]
        try:
            p=plain(proposal)
            if not isinstance(p,dict):raise ValueError('proposal must be an object')
            required={'capability_id','version','purpose','code','inputs','outputs','tools','expected_effects',
                'physical_side_effects','verification_refs','validation_status'}
            allowed=required|{'preconditions','applicability','known_limitations','semantic_dependencies','alternative_capabilities','output_projections'}
            if set(p)-allowed or required-set(p):raise ValueError('unknown or missing schema fields: '+str(sorted((set(p)-allowed)|(required-set(p)))))
            if p['validation_status']!='PROPOSED':raise ValueError('Only parent assigns maturity; proposal must be PROPOSED')
            if not all(isinstance(p[k],str) and p[k] for k in ('capability_id','version','purpose','code')):raise ValueError('nonempty id/version/purpose/code strings required')
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,79}',p['capability_id']):raise ValueError('invalid capability id')
            for key in ('inputs','outputs'):
                if not isinstance(p[key],dict) or len(p[key])>24 or not all(isinstance(k,str) and t in TYPES for k,t in p[key].items()):raise ValueError('invalid '+key+' type schema')
            for key in ('tools','expected_effects','physical_side_effects','verification_refs','semantic_dependencies','alternative_capabilities'):
                if not isinstance(p.get(key,[]),list) or not all(isinstance(v,str) for v in p.get(key,[])):raise ValueError(key+' must be string list')
            if not p['tools'] or set(p['tools'])-set(self.functions):raise ValueError('Unknown or missing real API binding')
            projections=p.get('output_projections',{})
            if not isinstance(projections,dict) or any(not isinstance(k,str) or output_path(k,p['outputs'])[0] not in p['outputs'] for k in projections):raise ValueError('invalid output projection fields')
            for rule in projections.values():
                if (not isinstance(rule,dict) or set(rule)-{'call','tool','path','operation'} or
                    type(rule.get('call')) is not int or rule['call']<0 or rule.get('tool') not in p['tools'] or
                    not isinstance(rule.get('path'),list) or len(rule['path'])>12 or
                    not all(type(x) in (str,int) for x in rule['path']) or
                    rule.get('operation','identity') not in ('identity','length')):
                    raise ValueError('invalid output projection: call index, tool, path and optional identity/length required')
            expected=set(x for name in p['tools'] for x in self.manifest[name]['physical_side_effects'])
            if set(p['physical_side_effects'])!=expected:raise ValueError('Side effects must equal actual bound API effects: '+str(sorted(expected)))
            known=self.condition_state()
            if set(p['expected_effects'])-set(known):raise ValueError('Claimed physical effects lack parent condition verifier')
            if set(p['verification_refs'])-({'return_schema'}|set(p['expected_effects'])):raise ValueError('Unknown verification reference')
            if set(p['expected_effects'])-set(p['verification_refs']):raise ValueError('Each effect needs its parent verification reference')
            tree=validate(p['code'])
            if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='capability':raise ValueError('Code must define only def capability(named_inputs)')
            fn=tree.body[0]
            if fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.args.defaults or fn.args.kwonlyargs or fn.args.posonlyargs or [a.arg for a in fn.args.args]!=list(p['inputs']):raise ValueError('Function signature must exactly match ordered input schema, no defaults/decorators')
            locals_={n.name for n in ast.walk(fn) if isinstance(n,ast.FunctionDef)}-{'capability'}
            for n in ast.walk(fn):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):
                    if n.func.id not in set(p['tools'])|PURE|locals_:raise ValueError('Unbound/recursive/dynamic call: '+n.func.id)
                    if n.func.id in self.functions:
                        try:inspect.signature(self.functions[n.func.id]).bind(*([None]*len(n.args)),**{k.arg:None for k in n.keywords})
                        except TypeError as exc:raise ValueError('API signature mismatch '+n.func.id+': '+str(exc)) from exc
            bindings={name:self.manifest[name] for name in p['tools']}
            key=p['capability_id']+'-'+digest([p,bindings])[:24]
            row=dict(id=key,proposal=p,bindings=bindings,status='VALIDATED',permission=False,physical_verified=False,
                semantic_edges=[dict(kind='SEMANTIC_DEPENDENCY',relation=field,target=target,verified=False) for field in ('semantic_dependencies','alternative_capabilities') for target in p.get(field,[])])
            path=self.root/(key+'.json')
            try:
                with path.open('x') as f:json.dump(row,f,indent=2)
            except FileExistsError:pass
            node=self.graph.add('capability',dict(id=key,proposal=p,semantic_edges=row['semantic_edges'],validation='interface_binding_only'))
            self.nodes[key]=node;self.event('registered',capability_id=key,node_id=node)
            qualification=self.qualify(key,p) if self.qualify else dict(status='CALLABLE_ONLY',reason='no parent composition compiler')
            self.event('planner_qualification',capability_id=key,qualification=qualification)
            return dict(id=key,status='VALIDATED',permission=False,node_id=node,planner_qualification=qualification)
        except (ValueError,TypeError,SyntaxError,KeyError) as exc:errors.append(str(exc))
        self.event('registration_rejected',errors=errors,proposal=proposal)
        return dict(status='REJECTED',errors=errors,permission=False,task_may_continue=True)

    def query(self,id=None):
        if id is None:
            return [{k:r.get(k) for k in ('id','status','physical_verified','verified_episodes')}|{'purpose':r['proposal']['purpose']} for r in [self.query(p.stem) for p in sorted(self.root.glob('*.json'))]][-40:]
        if not re.fullmatch(r'[A-Za-z0-9_-]+',id):return dict(status='UNKNOWN')
        path=self.root/(id+'.json')
        if not path.exists():return dict(status='UNKNOWN')
        row=json.loads(path.read_text());events=self.root/'events.jsonl';contexts=set()
        if events.exists():
            for line in events.read_text().splitlines():
                e=json.loads(line)
                if e.get('capability_id')==id and e['kind']=='execution':
                    row.update({k:e[k] for k in ('status','physical_verified','verification_scope','verified_episodes')})
                    if e['status'] in ('VERIFIED','REUSABLE') and e.get('verification_scope')!='return_schema_only':
                        contexts.add(digest([semantic_case(e.get('params',{})),[(c['name'],semantic_case(c.get('inputs')),semantic_case(c.get('output')),c.get('implementation_hash')) for c in e.get('calls',[])] ]))
        row['verified_contexts']=sorted(contexts)
        if row['status']=='REUSABLE':row.update(status='VERIFIED',maturity_note='reusability_requires_parent_declared_domain_validation')
        # Old type-only attestations remain in the append-only history, but must
        # never be promoted to semantic/physical verification by retrieval.
        if row.get('verification_scope')=='return_schema_only':
            row.update(status='EXECUTED',verification_scope='unverified',verified_episodes=[],
                       maturity_note='legacy_output_type_check_was_not_semantic_verification')
        if any(self.manifest.get(n)!=b for n,b in row['bindings'].items()):row.update(status='DEPRECATED',reason='implementation_or_contract_version_changed')
        return row

    def run(self,id,params,evidence_ids=None):
        row=self.query(id)
        if row['status'] in ('UNKNOWN','DEPRECATED'):return dict(status=row['status'],reason=row.get('reason'),permission=False)
        if self.running:return dict(status='REJECTED',reason='nested capability dispatch forbidden')
        p=row['proposal']
        if not check_values(params,p['inputs']):return dict(status='REJECTED',reason='Input type/key mismatch',expected=p['inputs'])
        evidence_ids=evidence_ids or []
        if not isinstance(evidence_ids,list) or not all(isinstance(x,str) and self.evidence_valid(x) for x in evidence_ids):return dict(status='REJECTED',reason='stale or unknown evidence')
        calls=[];returned=[];references={};before=copy.deepcopy(self.condition_state())
        node=self.graph.add('call',dict(capability_id=id,status='running',dataflow_completeness='explicit_refs_and_API_calls_only'),
            parents=[n for n in evidence_ids if n in self.graph.nodes]+([(self.nodes[id],'attestation')] if id in self.nodes else []))
        def callback(name,args,kwargs):
            if name=='capability_return':
                returned.append(plain(args[0]));return None
            call=dict(name=name,inputs=plain(dict(args=args,kwargs=kwargs)),
                code_site=copy.deepcopy(actor.call_site),implementation_hash=self.manifest[name]['implementation_hash'])
            input_refs=string_values([args,kwargs])
            parents=[references[ref] for ref in input_refs if ref in references]
            call_node=self.graph.add('call',dict(call,capability_execution=node,
                dataflow_completeness='explicit_return_ID_references_only'),parents=list(dict.fromkeys(parents))+[(node,'attestation')])
            call['call_node']=call_node
            calls.append(call)
            try:value=self.callback(name,args,kwargs)
            except Exception as exc:
                call.update(status='FAILED',error=str(exc))
                self.graph.invalidate([call_node],reason='actual bound API failed: '+str(exc));raise
            call.update(status='RETURNED',output=plain(value))
            self.graph.returned_call(call_node)
            call['result_node']=self.graph.add('tool_data',dict(api=name,output=plain(value),physical_verified=False),parents=[(call_node,'attestation')]+result_dependencies(value,self.graph),state='done')
            for ref in result_ids(plain(value)):references[ref]=call['result_node']
            return value
        functions={n:self.functions[n] for n in p['tools']};functions['capability_return']=lambda value:None
        code=p['code']+'\ncapability_return(capability(**'+repr(params)+'))'
        self.running=True;started=time.monotonic();actor=SandboxProcess(functions,callback=callback)
        try:
            result=actor.run(code,timeout=max(.01,min(120.,self.seconds())))
        finally:actor.close();self.running=False
        output=returned[0] if len(returned)==1 else None
        schema_ok=result['ok'] and check_values(output,p['outputs'])
        after=self.condition_state();effects=p['expected_effects']
        physical_verified=bool(effects) and schema_ok and all(after.get(k,{}).get('truth')=='TRUE' and after.get(k)!=before.get(k) for k in effects)
        projection_validation={}
        for field,rule in p.get('output_projections',{}).items():
            try:
                call=calls[rule['call']]
                if call['name']!=rule['tool'] or call['status']!='RETURNED':raise ValueError('bound call did not return')
                expected=call['output']
                for key in rule['path']:expected=expected[key]
                if rule.get('operation')=='length':expected=len(expected)
                actual=output
                for key in output_path(field,p['outputs']):actual=actual[key]
                projection_validation[field]=dict(matches=bool(schema_ok and actual==expected),
                    call=rule['call'],tool=rule['tool'],path=rule['path'],operation=rule.get('operation','identity'))
            except (IndexError,KeyError,TypeError,ValueError) as exc:
                projection_validation[field]=dict(matches=False,reason=str(exc))
        covered=[tuple(output_path(k,p['outputs'])) for k,v in projection_validation.items() if v['matches']]
        projections_ok=bool(p['outputs']) and schema_ok and all(x['matches'] for x in projection_validation.values()) and all(
            any(leaf[:len(prefix)]==prefix for prefix in covered) for leaf in output_leaves(output))
        verified=schema_ok and (physical_verified or (not effects and projections_ok))
        scope='parent_fresh_condition_verifiers' if physical_verified else 'audited_output_projections' if verified else 'unverified'
        episodes=set(row.get('verified_episodes',[]))
        if verified:episodes.add(self.episode)
        verification_context=digest([semantic_case(params),[(c['name'],semantic_case(c['inputs']),semantic_case(c.get('output')),c['implementation_hash']) for c in calls]])
        contexts=set(row.get('verified_contexts',[]))
        if verified:contexts.add(verification_context)
        status='VERIFIED' if verified else 'EXECUTED' if result['ok'] else 'FAILED'
        projected_parents=list(dict.fromkeys(calls[p['output_projections'][field]['call']]['result_node']
            for field,check in projection_validation.items() if check['matches']))
        output_node=self.graph.add('tool_data',dict(output=output,verification_scope=scope,physical_verified=physical_verified,
            dataflow_completeness='validated_projections_only_other_computations_untracked'),parents=projected_parents+[(node,'attestation')],state='done' if result['ok'] else 'invalid')
        record=dict(status=status,params=params,output=output,maturity_note='verification cases are not an applicability-domain certificate',physical_verified=physical_verified,verification_scope=scope,
            output_schema_valid=bool(schema_ok),projection_validation=projection_validation,
            verification_context=verification_context,verified_contexts=sorted(contexts),
            verified_episodes=sorted(episodes),calls=calls,actor=result,seconds=time.monotonic()-started,node_id=node,output_node=output_node,permission=False)
        self.graph.nodes[node]['data'].update(record)
        if result['ok']:self.graph.returned_call(node)
        else:self.graph.invalidate([node],reason='capability execution failed')
        self.event('execution',capability_id=id,**record)
        # Do not convert an interrupted physical call into a successful tool return.
        if not result['ok'] and any(self.manifest[n]['physical_side_effects'] for n in p['tools']):
            from .sandbox import ExecutionInterrupted
            raise ExecutionInterrupted('Registered program interrupted; consult actual broker state: '+str(result.get('stderr',result.get('error'))))
        return dict(result_ref=output_node,dependency_refs=[output_node],
            **{k:v for k,v in record.items() if k not in ('calls','actor','verified_episodes')})


SEMANTIC_API_DOC='''
Dynamic semantic capabilities (same parent permissions for all methods):
capability_tools(): actual bound API signatures, implementation hashes, side effects.
register_capability(proposal): propose a composed function and semantic contract.
Required fields: capability_id/version/purpose (strings), code (ONLY def capability
with ordered named inputs), inputs and outputs (dict field->string type, types:
string/number/integer/boolean/array/object/null), tools (actual API name list),
expected_effects (existing parent condition keys, normally [] for computations),
physical_side_effects (EXACT union from capability_tools for all bound tools),
verification_refs (return_schema and/or each expected effect key),
validation_status='PROPOSED'. Optional preconditions/applicability/known_limitations,
semantic_dependencies/alternative_capabilities (hypotheses, not physical facts).
Returns VALIDATED with id, or REJECTED with errors; fix errors without ending task.
run_capability(id, params, evidence_ids=[]): existing Sandbox and parent broker,
same budgets; no files/network/new permissions. Returns output and verification_scope.
capability_query(id=None): retrieve version or compact persisted catalog.
Return type checks alone produce EXECUTED, NEVER VERIFIED or physical success.
Optional output_projections can independently verify a compact result against
actual bound API outputs: {output_field:{call:zero_based_actual_call_index,
tool:actual_API_name,path:[dict_keys_or_list_indices],operation:'identity'|'length'}}.
Use normal Python to calculate output; parent separately checks every projection.
Output fields may be nested via dotted keys (e.g. summary.status). Every output
leaf must be covered; arrays can be checked as whole values. All projections must
match to obtain VERIFIED in audited_output_projections scope.
This proves faithful extraction only, NOT usefulness, completeness or physical safety.
The parent alone verifies physical conditions. Successful cases remain VERIFIED
only in their recorded scope; multiple episodes do not certify an applicability
domain and do not automatically make a capability REUSABLE.
You may register a computation or combination useful for this task in the SAME
Python response. Do not register every call. Do not invoke nonexistent tools.
'''
