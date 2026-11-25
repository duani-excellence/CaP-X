"""Real DeepSeek decisions on archived PUBLIC evidence, not physical recovery.

Uses the existing provider budget, Sandbox and V35 IK search. No simulator,
hidden state, synthetic model response or physical dispatcher is constructed.
"""
import argparse,copy,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from capx.execution.agent_events import plain,digest
from capx.execution.sandbox import SandboxProcess
from capx.execution.dependency_graph import DependencyGraph
from capx.execution.semantic_capabilities import CapabilityRegistry,SEMANTIC_API_DOC
from capx.execution.decision_state import encode_decision_state
from capx.execution.preplanning import DiagnosticReachability,search_placements,planning_summary
from capx.execution.bounded_ik import BoundedIK,IKSolveBudget
from capx.execution.control_frames import transform_target
from capx.integrations.motion.pyroki import init_pyroki
from capx.experiments.graph_campaign import PersistentModelBudget
from capx.experiments.layered_feedback import validated_program
from scripts.local.diagnose_region_memory import load_frame
from capx.utils.request_audit import RequestAudit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--ik-runtime',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True);p.add_argument('--mode',choices=['legacy','encoded','semantic'],required=True)
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--target',required=True)
    p.add_argument('--prior-request',type=Path);p.add_argument('--prior-actor',type=Path)
    p.add_argument('--replay-program',type=Path,help='Replay unchanged archived program JSON; ZERO new LLM requests')
    a=p.parse_args()
    if not 1<=a.rounds<=3:raise ValueError('1..3 requests per bounded gate')
    if a.replay_program and (a.rounds!=1 or a.prior_request):raise ValueError('Replay needs one round and no prior-request')
    a.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (a.output/name).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2))
    d=json.loads(a.diagnosis.read_text());ep=Path(d['reference']);cfg=json.loads(a.config.read_text())
    context=json.loads(sorted((ep/'harness').glob('checkpoint-*.json'))[-1].read_text())['state']
    source=context['robot']['source'] if 'source' in context['robot'] else None
    frames=sorted((ep/'observations').glob('*-calibration.json'));index=int(frames[-1].name.split('-')[0])
    frame,source=load_frame(ep,index)
    cal=json.loads((ep/'calibration.json').read_text());urdf=next(x['path'] for x in cal['sources'] if x['role']=='ik_urdf')
    ref=np.r_[frame['robot_joint_pos'][:7],0.]
    def target(p,q):
        pos,quat=transform_target(cal,p,q)
        return np.r_[quat,pos+Rotation.from_quat(np.roll(quat,-1)).apply(cal['api_tcp_to_hand_offset_m'])]
    solver=BoundedIK(init_pyroki(json.loads(a.ik_runtime.read_text())['url']),urdf_path=urdf,budget=IKSolveBudget(512))
    ik=DiagnosticReachability(solver,target,lambda:ref,calibration=cal['calibration_hash'])
    prior=json.loads((a.diagnosis.parent/'ik-results.json').read_text())
    # Keep historical solver results queryable, but do not seed caches across a
    # changed solver reference. Exact new search keys remain independently checked.
    records=dict(prior);r=copy.deepcopy(d['result']);r.update(id='historical-placement',skill='deposit',object=d['object'],target=a.target)
    searches=[r];records[r['id']]=r
    skills=SimpleNamespace(held=d['attachment'],scene=SimpleNamespace(obs=frame,source=source),phase=context['phase'],
        profile=json.loads((ep/'error-profile.json').read_text()),reachable=ik)
    state={k:context[k] for k in ('task_goals','remaining_goals','conditions')}
    state['available_bindings']=[dict(skill='deposit',object=d['object'],target=r['target'])]
    belief=context['relevant_belief'];calls=[];outputs=[];search_cache={}
    def decision_state():
        """Compact decision state from frozen public measurements, never a permit."""
        return encode_decision_state(state,skills,belief,searches,{'physical_dispatch':0,'IK_iterations':512-solver.budget.calls},[context.get('failed_api')])
    def read_robot():
        """Archived encoders, TCP XYZ WXYZ and jaws. No fresh capture in this gate."""
        return dict(source=source,kind='archived_public_measurement',tcp=plain(frame['robot_cartesian_pos'][:7]),
            joints=plain(frame['robot_joint_pos']),gripper=plain(frame['robot_gripper_joint_pos']))
    def planning_query(id=None):
        """Query search / IK IDs, or list {id,status} with None.
        Search result fields: id,status,skill,object,target,tested (list of actual
        candidates with position,quaternion_wxyz,status,checks). Each checks entry
        has position_error_m,orientation_error_deg,status,failure_type if measured.
        No tested_summary or untested_scope fields: those belong to decision_state
        plans, not this full query. Missing values mean UNKNOWN, not zero.
        """
        all_records={**records,**ik.records}
        if id is None:return [{'id':k,'status':v['status']} for k,v in all_records.items() if k.startswith('search') or k=='historical-placement']
        return copy.deepcopy(all_records.get(id,dict(status='UNKNOWN')))
    def preview_skill(skill,params=None):
        """Actual bounded live IK search on frozen evidence. No physical actions.
        Only deposit with params object,target and optional preferred_xy or upright
        quaternion_wxyz is available. Other skills require new physical evidence.
        """
        params=params or {}
        if skill!='deposit' or params.get('object')!=d['object'] or str(params.get('target'))!=r['target'] or set(params)-{'object','target','preferred_xy','quaternion_wxyz'}:
            return dict(status='UNAVAILABLE',reason='Frozen gate exposes deposit planning only',permission=False)
        key=digest(params)
        if key in search_cache:return dict(planning_summary(search_cache[key]),cache_hit=True,new_solver_calls=0)
        out=search_placements(skills,params,d['attachment']['shape'],d['registration'],d['geometry'])
        out.update(id='search-'+key,skill=skill,object=d['object'],target=params['target'])
        search_cache[key]=out;records[out['id']]=out;searches.append(out)
        return planning_summary(out)
    functions={'read_robot':read_robot,'decision_state':decision_state,'planning_query':planning_query,'preview_skill':preview_skill}
    graph=DependencyGraph();registry=CapabilityRegistry(a.output.parent/'semantic-registry',functions,graph=graph,
        tool_effects={k:[] for k in functions},episode=str(a.output),condition_state=lambda:state['conditions'])
    if a.mode=='semantic':functions.update(capability_tools=lambda:registry.manifest,
        register_capability=registry.register,run_capability=registry.run,capability_query=registry.query)
    def dispatch(name,args,kwargs):
        start=time.monotonic();value=functions[name](*args,**kwargs)
        calls.append(dict(name=name,args=plain(args),kwargs=plain(kwargs),result=plain(value),seconds=time.monotonic()-start))
        return value
    registry.callback=dispatch
    tools='\n'.join(name+str(__import__('inspect').signature(fn))+': '+(fn.__doc__ or '') for name,fn in functions.items())
    facts=decision_state()
    # Same relevant facts, two representations; raw detailed data remains equally
    # available through the same tools. No oracle labels are supplied to either.
    if a.mode=='legacy':packet=dict(public_state=state,robot=read_robot(),belief={k:v for k,v in belief.items() if k in ('object:'+d['object'],'attachment:'+d['object'],'region:destination')},recent_search=facts['plans'],failed_api=context.get('failed_api'))
    else:packet=facts
    base=[dict(role='system',content='You are a robot Coding Agent. This is an OFFLINE DECISION AND PLANNING experiment with archived public RGB-D evidence and live IK. No physical robot actions are available. Do not claim physical recovery. Use only documented APIs and bounded Python; no filesystem/network/private attributes. Reserved variables env, obs, object are forbidden. No arbitrary methods; use get/items/keys/values/append/extend/tolist and numpy operations.'),
        dict(role='user',content=tools+'\n'+(SEMANTIC_API_DOC if a.mode=='semantic' else '')+'\nFrozen public failure state:\n'+json.dumps(plain(packet),ensure_ascii=False)+
             ('\nCreate and register a genuinely useful new capability combining at least two of the actual APIs, then execute it on this failure. Keep outputs compact. Do not claim a physical effect for an analysis function. Return ONE fenced Python program.' if a.mode=='semantic' else
              '\nDiagnose the failed stage and propose a materially different legal strategy from the tested search. Query details, then make at most one justified planning search if useful. Distinguish missing capability from no solution in tested set. Print a compact diagnosis and next action; return ONE fenced Python program.'))]
    budget=PersistentModelBudget(a.ledger,cfg['budgets']);startcount=budget.count;messages=base;actors=[]
    if a.prior_request:
        previous=json.loads((a.prior_request/'response.json').read_text())
        messages=json.loads((a.prior_request/'prompt.json').read_text())+[
            dict(role='assistant',content=previous['content']),
            dict(role='user',content='Actual Runtime rejection from the preceding attempt:\n'+a.prior_actor.read_text()+
                 '\nCurrent shared API contracts (supersede old contract text):\n'+tools+'\n'+SEMANTIC_API_DOC+
                 '\nCurrent public decision packet:\n'+json.dumps(plain(packet),ensure_ascii=False)+
                 '\nCorrect the contract and execute it. Function inputs must have exactly the declared names and no default values; pass every declared input. Verification refs are strings. Use output_projections for faithful fields extracted from at least two real tools; do not equate extraction with physical recovery. Return one Python program.')]
    save('input.json',dict(mode='frozen_public_decision_live_DeepSeek_NOT_physical_recovery',encoding=a.mode,source=source,packet=packet,tool_manifest=registry.manifest))
    with RequestAudit(a.output/'tool-http.jsonl'):
        for i in range(a.rounds):
            if a.replay_program:
                original=json.loads(a.replay_program.read_text());code=original['code'];path=original.get('request')
                answer=dict(content='```python\n'+code+'\n```');origin='archived_model_program_ZERO_new_LLM'
            else:
                answer,path=budget.query(messages,a.output.name+'-'+str(i));code=validated_program(answer);origin='real_deepseek'
            save(f'program-{i}.json',dict(code=code,request=path,origin=origin,replay_reference=str(a.replay_program) if a.replay_program else None))
            actor=SandboxProcess(functions,callback=dispatch)
            try:result=actor.run(code,timeout=240)
            finally:actor.close()
            actors.append(result);save(f'actor-{i}.json',result)
            messages=messages+[dict(role='assistant',content=answer['content']),dict(role='user',content='Actual Sandbox/tool result:\n'+json.dumps(plain(result))+'\nTool outcomes and available result IDs:\n'+json.dumps([dict(name=c['name'],status=c['result'].get('status') if isinstance(c['result'],dict) else None,id=c['result'].get('id') if isinstance(c['result'],dict) else None) for c in calls[-6:]])+'\nCorrect any contract error and continue with a bounded, substantive new step. If the supported capability is exhausted, explicitly explain it. Return one Python program.')]
    save('calls.json',calls);save('planning-results.json',records|ik.records);save('graph.json',dict(nodes=graph.nodes,edges=graph.edges,events=graph.events))
    save('result.json',dict(mode='frozen_public_archived_model_replay' if a.replay_program else 'frozen_public_decision_live_DeepSeek_NOT_physical_recovery',physical_calls=0,
        new_llm_requests=budget.count-startcount,actors=actors,API_calls=len(calls),IK_iterations=solver.budget.calls,cache_hits=ik.hits,
        semantic_capabilities=registry.query(),input_characters=len(json.dumps(packet)),physical_success_tested=False))
    print(json.dumps(dict(mode=a.mode,requests=budget.count-startcount,actors=[x['status'] for x in actors],API_calls=len(calls),IK_iterations=solver.budget.calls)))

if __name__=='__main__':main()
