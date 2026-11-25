"""Official LOGIV executive on CaP-X public Vision Skill callbacks.

This is an adapted method, NOT the paper's VLA/GPT-4o/FastDownward/VAL
reproduction. Its execution, verifier, causal graph and Repairer are imported
unchanged from a revision-pinned checkout. Common finite capability search and
symbolic validation replace the external PDDL binaries; never import truth.
"""
import copy
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid


def load_official(path,revision):
    path=Path(path).resolve()
    actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:raise ValueError('LOGIV checkout revision mismatch')
    files=['logiv/core/'+x+'.py' for x in ['symbolic','graph','repair','execution','verification']]
    files+=['logiv/planning/solver.py','logiv/planning/pddl.py']
    for f in files:
        expected=subprocess.check_output(['git','-C',str(path),'show',revision+':'+f])
        if (path/f).read_bytes()!=expected:raise ValueError('Modified official LOGIV source: '+f)
    if 'logiv' in sys.modules and not str(Path(sys.modules['logiv'].__file__).resolve()).startswith(str(path)+'/'):
        raise ValueError('Different LOGIV module already imported')
    sys.path.insert(0,str(path))
    from logiv.core.execution import LOGIV
    from logiv.core.verification import Verifier,Observation
    from logiv.core.symbolic import State,Operator,Literal,regress
    from logiv.planning.solver import Validation,plan_key
    return dict(locals(),hashes={f:hashlib.sha256((path/f).read_bytes()).hexdigest() for f in files})


def execute_official(runtime,boundary,initial_code,config,source):
    from capx.execution.causal_plan import program_order,ground_library,required,changes,signature
    from capx.execution.sandbox import SandboxProcess
    from capx.experiments.graph_v2 import save
    api=load_official(source['path'],source['revision'])
    State,Operator,Literal=api['State'],api['Operator'],api['Literal']
    Observation=api['Observation'];Validation=api['Validation']
    c=runtime.c;initial_ids=program_order(initial_code,c.intents);rows={}
    # LOGIV world literals describe physical facts. CaP-X tool products (e.g.
    # an empty-hand grasp plan) are transient computations, not world invariants
    # that must stay observable while holding another object. The unchanged
    # common broker still reacquires and checks ALL of them before dispatch.
    tool_data={k for k,v in c.definition.get('condition_specs',{}).items() if v.get('kind')=='tool_data'}
    def operator(row):
        key='capability_'+signature(row);rows[key]=copy.deepcopy(row)
        return Operator(key,(),frozenset(Literal((k,),v=='TRUE') for k,v in required(row).items() if k not in tool_data),
                        frozenset(Literal((k,),v=='TRUE') for k,v in changes(row).items() if k not in tool_data),
                        frozenset(row.get('resources',[])),instruction=row['skill'])
    initial=tuple(operator(c.intents[i]) for i in initial_ids)
    library=tuple(operator(r) for r in ground_library(c.definition))
    initial_atoms={literal.atom for op in initial for literal in op.preconditions}
    capability_atoms=frozenset(literal.atom for op in library for literal in op.preconditions
                               if literal.atom not in initial_atoms)
    goal=frozenset(Literal((k,)) for k in c.definition['goals']);events=[];actors=[];counter=0
    class Validator:
        def validate(self,current,plan,goal):
            for i,op in enumerate(plan):
                if not current.holds(op.preconditions):return Validation(False,i,'missing_precondition')
                current=op.apply(current)
            return Validation(current.holds(goal),detail='common_symbolic_validation_no_physical_truth')
    class Planner:
        first=True
        def plan(self,current,goal,excluded=frozenset()):
            if self.first:
                self.first=False;return initial
            start=time.monotonic();frontier=[(0.,0,0,current,())];serial=itertools.count(1);seen=set();count=0
            while frontier and count<512:
                cost,depth,_,now,plan=heapq.heappop(frontier);key=(now.facts,now.unknown)
                if key in seen:continue
                seen.add(key);count+=1
                if now.holds(goal) and api['plan_key'](plan) not in excluded:
                    events.append(dict(kind='official_repair_search',expansions=count,skills=[rows[x.name]['skill'] for x in plan],seconds=time.monotonic()-start));return plan
                if depth>=6:continue
                for op in library:
                    if not now.holds(op.preconditions):continue
                    nxt=op.apply(now)
                    if nxt==now:continue
                    heapq.heappush(frontier,(cost+rows[op.name].get('cost',1.),depth+1,next(serial),nxt,plan+(op,)))
            events.append(dict(kind='official_repair_search_exhausted',expansions=count,seconds=time.monotonic()-start));return None
    class Runtime:
        def observe(self,active=False):return Observation(robot={'phase':runtime.skills.phase},raw={'active':active})
        def execute(self,op,observation):
            nonlocal counter
            row=copy.deepcopy(rows[op.name]);old=row.get('id')
            if old not in c.intents or old in c.completed or old in c.stopped:
                row['id']='external-'+uuid.uuid4().hex;row['after']=[];c.register(row)
            intent=row['id'];runtime.last_intent=intent
            boundary.functions={'execute_skill':runtime.broker.execute}
            boundary.actor.close();boundary.actor=SandboxProcess(boundary.functions,callback=boundary.call)
            code=f'execute_skill({intent!r})';save(boundary.root/f'actor-{counter:02d}-program.json',dict(code=code,origin='official_LOGIV_selected_skill',shared_initial_program=True))
            started=time.monotonic();result=boundary.actor.run(code,timeout=min(config['budgets']['block_seconds'],runtime.broker.deadline-time.monotonic()))
            result['seconds']=time.monotonic()-started;actors.append(result);save(boundary.root/f'actor-{counter:02d}-result.json',result);counter+=1
            events.append(dict(kind='official_dispatch',intent=intent,skill=row['skill'],status=result['status'],seconds=result['seconds']))
            runtime._sync_stages()
            if not result['ok']:runtime.last_error=result.get('error') or result.get('stderr') or result['status']
            if c.quarantined:raise RuntimeError('Unknown physical effect quarantined')
            return self.observe()
    class Perception:
        def assess(self,observation,atoms):
            # The same current public verifier is used in every method. No
            # predicted effect is substituted for a missing measurement.
            keys=[a[0] for a in atoms];observed=runtime.broker.refresh(keys)
            events.append(dict(kind='official_verification',conditions={k:v['truth'] for k,v in observed.items()}))
            return {a:({'TRUE':True,'FALSE':False}.get(observed[a[0]]['truth'])) for a in atoms}
    # The official compiler requires symbolic plan support at preparation.
    # Obtain that support from public perception; never seed it as TRUE.
    pre=api['regress'](goal,initial)
    if pre is None:raise ValueError('Official LOGIV rejected inconsistent initial program')
    runtime.broker.refresh([x.atom[0] for x in pre]);runtime._sync_stages()
    # Keep UNKNOWN for exactly facts queried so far. Future unread facts are
    # not required by the compiler; action gates always query before dispatch.
    initial_state=State(frozenset((k,) for k in c.conditions if k not in tool_data and c.truth(k)=='TRUE'),
                        frozenset((k,) for k in c.conditions if k not in tool_data and c.truth(k)=='UNKNOWN'))
    # Repair-library conditions not used by the initial plan must still enter
    # the official planner's CURRENT state (e.g. a newly stalled approach).
    # Use LOGIV's own tracked_atoms facility with common public verification.
    engine=api['LOGIV'](Planner(),Validator(),api['Verifier'](Runtime(),Perception(),additional_observations=2,tracked_atoms=capability_atoms),
                        execution_budget=config['tasks']['pick_place']['skill_budget'],preparation_attempts=1)
    result=engine.run(initial_state,goal)
    save(boundary.root/'external-method.json',dict(method='LOGIV-official-core-common-skills-adapter',revision=source['revision'],
        source_hashes=api['hashes'],outcome=result.outcome.value,reason=result.reason,rounds=result.rounds,
        completed=[rows[x.name]['skill'] for x in result.completed],failures=[str(x) for x in result.failures],events=events,
        adaptations=['public Vision Skill verifier replaces GPT-4o/VLA feedback','finite common library search replaces FastDownward','symbolic validator replaces VAL','transient tool data stays in common broker gates, not persistent LOGIV world literals'],
        transient_tool_conditions=sorted(tool_data),all_tool_preconditions_enforced_by_common_broker=True,
        tracked_capability_conditions=sorted(a[0] for a in capability_atoms),
        original_paper_reproduction=False,training_tokens=0,training_cost=0))
    return actors,'external_LOGIV_'+result.reason
