"""Public Python recovery tools; all physical dispatch stays in the parent.

This version exposes the existing certified vertical probe corridor, not an
unqualified arbitrary Cartesian planner. No raw release or simulator handles.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .agent_events import plain,digest

OPEN_API_DOC='''V34 OPEN Code-as-Policy API, shared by initial and recovery programs:
Write one complete fenced Python program. Calculations, dictionaries, conditionals,
functions and bounded loops (e.g. range(4)) are allowed by the existing Sandbox.
No imports required. env, obs and globals are RESERVED: forbidden even as local
variable names. Use frame/data instead. No private attributes, files or network.
Allowed methods: get/items/keys/values/copy/append/extend/tolist/reshape/astype.
No arbitrary attributes, startswith, sort or exception handling around safety stops.
All poses: robot_base XYZ metres, unit quaternion WXYZ, robot TCP (not hand origin).
execution_state(): {conditions: {key: {truth:'TRUE'|'FALSE'|'UNKNOWN',node,evidence}},
remaining_goals:[string],attachment:object_name_or_None,phase:string,intents:[...] }.
Read condition truth with state['conditions'].get(key, {}).get('truth'); do NOT
compare the condition object itself to a string.
read_robot(): fresh {id, tcp:[x,y,z,w,x,y,z], robot_joint_pos,
robot_gripper_joint_pos, wrist_ft if enabled}.
observe_object(object): current public measured geometry with evidence id; UNKNOWN is not absence.
observe_grasp(object): validated grasp proposal {id,position,quaternion_wxyz}; no motion.
observe_surfaces(role): role='source' or 'destination', current fit or UNKNOWN with
axis, wall_diagnostics and history_check. Historical surfaces must pass fresh depth tests.
plan_skill(skill, inputs={}, params={}): no motion. Exact success status is
READY_FOR_LIVE_VERIFICATION, with id, requires and permission=False. To execute:
if plan.get('status') == 'READY_FOR_LIVE_VERIFICATION': run_plan(plan['id'])
Other statuses: ALREADY_SATISFIED (no id) and REJECTED (no id). Never test for 'ok'.
Skills: acquire, deposit, confirm_attachment, retreat_empty. params must include object;
retreat_empty ALSO requires params={'object': the_interrupted_object}; it is a
qualified retreat after an interrupted empty approach, not a general post-place reset.
deposit also requires target (string cell index from task binding), may include preferred_xy
and quaternion_wxyz (upright yaw). acquire may use candidate_id from preview_skill
selected.candidate_id; execution rechecks the SAME grasp, never silently substitutes.
acquire also accepts future_positions (integer 1..12, default 2): a bounded
number of placement positions per upright yaw tested for each of up to 3 grasps.
If finite search fails, inspect planning_query(id).joint_search and future_search;
increasing this bound searches more candidates, never relaxes motion/release gates.
inputs contains source IDs returned by observe APIs.
Contracts, object binding, current conditions, resource locks and safety remain parent-owned.
inputs is optional: mature skills gather and validate their own required observations.
Do not preobserve every object/region before every skill. Read only missing evidence.
Explicit inputs accept ONLY id returned by observe_*, read_robot or plan_* tools,
not arbitrary IDs from graph_query. Stale inputs must not be reused after motion.
You choose which object/skill and order. No fixed execute_skill sequence is required.
run_plan(plan_id): executes a NEW parent-validated plan, returns verified effects or
interrupts at an unsafe/uncertain boundary. Never re-run a completed plan. Every actor
has a fresh Python state but physical state and prior plans/effects persist.
plan_motion(position, quaternion_wxyz, inputs={}, purpose='observation'): proposes a
bounded Cartesian path, accepts lateral motion and orientation changes, up to 0.15m
and 30deg per plan. Subsegments, fresh RGB-D swept tool/attached volume, actual IK
joint-sweep, convergence, held verification and F/T guards are mandatory. UNKNOWN
space rejects the plan. This is local sampled planning, not arbitrary global planning.
Use read_robot() and geometric diagnostics to propose a useful viewpoint, not random jiggles.
move_tcp(position, quaternion_wxyz, operation_id): plan+run shorthand, same checks.
recall_result(id): retrieve still-valid public result or explicit stale error.
belief_query(key=None, history=False): structured public estimates independent of
condition truth; keys include region:destination, object:<name>, attachment:<name>,
robot and quality:<name>. These are estimates, never execution permits.
history_query(version): retrieve one prior program and its actual origin, on demand.
observation_opportunities(role): estimate missing-wall occlusion by current TCP
from calibrated fixed cameras; suggestions are NOT motion permissions. A fixed
camera does not move when the TCP moves. Query only if that geometry is missing.
Return schema: {status:'CANDIDATES'|'UNKNOWN',candidates:[{position:[x,y,z],
quaternion_wxyz:[w,x,y,z],intent,permission:False}],reason,permission:False}.
capture_observation(), inspect_region(role), verify_conditions(list_of_keys),
graph_query('state'|'failures'|'evidence'|'dependencies', key=None) also available.
Existing execute_skill, repair_skill and refresh_evidence remain compatibility tools.
No raw release/open_gripper: deposit retains the original final release verifier.
Plan rejection returns diagnostic information so you can try another meaningful plan;
physical safety stops interrupt the actor and require current state reconciliation.
Preserve completed goals. New source IDs, tiny motion and repeating a failed strategy
are not progress. FINISH or normal Python return is not physical task success.
'''


API_DOC='''Additional V33 public Python APIs (both G0 and G1):
capture_observation(): new RGB-D capture; returns source refs, robot.tcp (xyz+wxyz),
joints, gripper encoders, optional wrist_ft, image identities and motion_scope.
No image file access or private simulator state. DeepSeek receives structured data.
inspect_region(role): role is source or destination; current public geometric fit
or {status:UNKNOWN, reason}. It does not treat historic bounds as current truth.
verify_conditions(keys): current public verification; returns per-key truth/evidence.
move_tcp(position, quaternion_wxyz, operation_id): parent-checked observation motion.
ONLY a verified held object inside its original observed vertical approach corridor
is supported: same XY/orientation, upward 0.001..0.02 m per call, target at most
original grasp Z + 0.20 m. Requires fresh held verification, current neighbor and
receptacle sweep, IK, TCP convergence and F/T guard. Other motion is rejected.
Use current robot.tcp and motion_scope; never infer an unlisted camera/arm API.
operation_id is a fresh descriptive string: completed IDs cannot be replayed.
graph_query(query, key=None): query is state, dependencies, evidence, or failures.
The graph is advice/memory, not authority. Conditions never become TRUE by query.
Programs may combine these APIs, calculations, existing execute_skill and
repair_skill calls. No raw open_gripper/release is available; deposit retains all
current release checks. Newly observed conditions may enable existing remaining
intents without a graph-work-list lock, but parent contracts still apply.
No task success can be asserted by print/FINISH. If no supported safe new strategy
exists, explain the missing capability. Do not repeat an equivalent observation.
Python sandbox restrictions from the initial API documentation still apply.
'''


def admitted_observation_target(tcp,origin,position,quaternion):
    tcp=np.asarray(tcp,float);origin=np.asarray(origin,float);p=np.asarray(position,float);q=np.asarray(quaternion,float)
    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all() or abs(np.linalg.norm(q)-1)>1e-3:raise ValueError('Invalid pose')
    angle=np.degrees((Rotation.from_quat(np.roll(tcp[3:7],-1)).inv()*Rotation.from_quat(np.roll(q,-1))).magnitude())
    if np.linalg.norm(p[:2]-tcp[:2])>1e-5 or angle>1e-3:raise ValueError('Only current certified vertical observation corridor; lateral/rotating motion unsupported')
    if not .001<=p[2]-tcp[2]<=.020001 or p[2]>origin[2]+.20+1e-6:raise ValueError('Observation target outside bounded vertical corridor')
    return p,q


class OpenRecoveryAPIs:
    def __init__(self,runtime,store,open_runtime=False):
        self.r=runtime;self.s=runtime.skills;self.store=store;self.operations=set();self.last_observation=None
        self.s.validation_sink=self.reconcile_validation
        self.open_runtime=open_runtime;self.results={};self.plans={};self.information={};self.search_results=[];self.strategies={}
        if open_runtime:
            self.s.open_runtime=True;self.s.scene.allow_region_memory=True
            if hasattr(self.s,'preplanner'):
                self.s.preplanner.targets={r['object']:r['target'] for r in self.r.library if r['skill']=='deposit'}

    def reconcile_validation(self,effects,reason='trusted_verifier'):
        # Only trusted parent validators call this method. No actor RPC exposes it.
        for key,row in effects.items():
            self.r.c.observe(key,row['truth'],row.get('evidence',{}))
            self.s.effects[key]=row['truth']
            if self.s.held and key=='held:'+self.s.held['object'] and row['truth']!='TRUE':
                self.s.phase='closed_unverified'
                if hasattr(self,'belief'):
                    self.belief.record('attachment:'+self.s.held['object'],dict(status='UNKNOWN',reason=reason),source=self.s.scene.source)
        self.store.append('VerificationReconciled',dict(effects=effects,reason=reason))

    def belief_query(self,key=None,history=False):
        return self.belief.query(key,history)

    def history_query(self,version):
        if type(version) is not int or version<0:raise ValueError('Prior program version required')
        path=self.store.root/f'program-{version:03d}.json'
        if not path.is_file():return dict(status='UNKNOWN',reason='no such prior program')
        import json
        return json.loads(path.read_text())

    def capability_tools(self):
        return copy.deepcopy(self.capabilities.manifest)

    def register_capability(self,proposal):
        return self.capabilities.register(proposal)

    def run_capability(self,id,params,evidence_ids=None):
        return self.capabilities.run(id,params,evidence_ids)

    def capability_query(self,id=None):
        return self.capabilities.query(id)

    def observation_opportunities(self,role):
        from .open_motion import observation_opportunities
        if role not in ('source','destination'):raise ValueError('Unknown region role')
        self._capture()
        result=observation_opportunities(self.s,role)
        self.store.append('InformationActionCandidates',result)
        return result

    def _result(self,kind,data,parents=()):
        c=self.r.c
        monitor=getattr(self,'monitor',None);call=[(monitor.call_stack[-1],'attestation')] if monitor and monitor.call_stack else []
        node=c.graph.add('estimate',dict(type=kind,**plain(data)),parents=list(parents)+call+([(c.program_node,'attestation')] if c.program_node else []))
        self.results[node]=dict(type=kind,**plain(data));return dict(id=node,result_ref=node,dependency_refs=[node],**plain(data))

    def _inputs(self,inputs):
        if not isinstance(inputs,dict) or not all(isinstance(v,str) for v in inputs.values()):raise ValueError('Input evidence IDs required')
        for value in inputs.values():self.recall_result(value)
        return list(inputs.values())

    def recall_result(self,id):
        if id not in self.results or not self.r.c.graph.nodes[id]['valid']:raise ValueError('Unknown or stale result ID; reobserve only necessary evidence')
        return dict(id=id,result_ref=id,dependency_refs=[id],**copy.deepcopy(self.results[id]))

    def motion_changed(self):
        if hasattr(self,'belief'):self.belief.robot_motion(self.s.scene.source)
        for node,row in self.results.items():
            if row['type'] in ('robot','grasp','motion_plan') or (row['type']=='object' and self.s.held and row.get('object')==self.s.held['object']):
                self.r.c.graph.invalidate([node],reason='measured robot or attachment pose changed')

    def _capture(self):
        if self.r.refreshes>=self.r.limit:raise ValueError('Shared evidence budget exhausted')
        self.r.refreshes+=1;self.s.scene.frame()

    def read_robot(self):
        self._capture();v=self.observation()
        return self._result('robot',dict(v['robot'],source=self.s.scene.source,motion_version=self.s.motion_version))

    def observe_surfaces(self,role):
        value=self.inspect_region(role)
        return self._result('region',dict(value,role=role))

    def observe_object(self,object):
        if object not in self.s.scene.bindings:raise ValueError('Unknown public object binding')
        self._capture()
        try:
            if not self.s.anchors:self.s.prepare(capture=False)
            if self.s.held and self.s.held['object']==object:
                value,reg=self.s.current_held();self.information[object]=reg
            else:value=self.s.scene.object(object,self.s.region('source'))
            return self._result('object',dict(status='PASS',object=object,source=self.s.scene.source,
                geometry={k:v for k,v in value.items() if k not in ('points','views')}))
        except (ValueError,RuntimeError) as exc:
            from .skill_contracts import SkillBoundaryStop
            if not isinstance(exc,(ValueError,SkillBoundaryStop)):raise
            if isinstance(exc,SkillBoundaryStop):self.reconcile_validation(exc.effects,str(exc))
            return self._result('object',dict(status='UNKNOWN',object=object,reason=str(exc),source=self.s.scene.source))

    def observe_grasp(self,object):
        if object not in self.s.scene.bindings:raise ValueError('Unknown object')
        self._capture()
        if not self.s.anchors:self.s.prepare(capture=False)
        try:
            g,p,q,roles=self.s._acquire_plan(object)
            self.s.plans[object]=(self.s.scene.serial,(g,p,q,roles))
            return self._result('grasp',dict(status='PASS',object=object,position=p,quaternion_wxyz=q,
                source=self.s.scene.source,motion_version=self.s.motion_version))
        except (ValueError,RuntimeError) as exc:
            from .skill_contracts import SkillBoundaryStop
            if not isinstance(exc,(ValueError,SkillBoundaryStop)):raise
            if isinstance(exc,SkillBoundaryStop):self.reconcile_validation(exc.effects,str(exc))
            return self._result('grasp',dict(status='UNKNOWN',object=object,reason=str(exc),source=self.s.scene.source))

    def preview_skill(self,skill,params=None):
        if not hasattr(self.s,'preplanner'):return dict(status='UNAVAILABLE',permission=False)
        params=params or {}
        allowed={'object','target','preferred_xy','quaternion_wxyz','candidate_id','future_positions'}
        if not isinstance(params,dict) or set(params)-allowed:raise ValueError('Unsupported planning parameter')
        if params.get('object') not in self.s.scene.bindings:raise ValueError('Unknown object binding')
        if not any(r['skill']==skill and r.get('object')==params['object'] and
            (skill!='deposit' or r.get('target')==str(params.get('target'))) for r in self.r.library):
            return dict(status='REJECTED',reason='No qualified binding',permission=False)
        self._capture()
        if not self.s.anchors:self.s.prepare(capture=False)
        row=dict(skill=skill,**params)
        previous=getattr(self.s,'requested_grasp_candidate',None)
        self.s.requested_grasp_candidate=params.get('candidate_id')
        try:result=self.s.preplanner.preview(row)
        finally:self.s.requested_grasp_candidate=previous
        self.search_results.append(dict(strategy=skill,parameters=params,outcome=result['status'],
            candidate=result.get('selected'),id=result['id']))
        self.store.append('PlanningResult',result)
        from .preplanning import planning_summary
        return planning_summary(result)

    def planning_query(self,id=None):
        planner=getattr(self.s,'preplanner',None)
        if planner is None:return dict(status='UNAVAILABLE')
        if id is None:return [dict(id=r['id'],status=r['status'],skill=r['skill'],object=r.get('object')) for r in list(planner.records.values())[-12:]]
        if id in planner.records:return copy.deepcopy(planner.records[id])
        if hasattr(self.s.reachable,'records') and id in self.s.reachable.records:return copy.deepcopy(self.s.reachable.records[id])
        return dict(status='UNKNOWN',reason='Unknown planning result')

    def decision_state(self):
        from .decision_state import encode_decision_state
        planner=getattr(self.s,'preplanner',None)
        state=dict(self.r.state(),task_goals=self.r.c.definition['goals'],
            available_bindings=[{k:r.get(k) for k in ('skill','object','target','requires')} for r in self.r.library])
        return encode_decision_state(state,self.s,self.belief.query(),
            list(planner.records.values()) if planner else [],{},self.r.events[-3:])

    def plan_skill(self,skill,inputs=None,params=None):
        import uuid
        inputs=inputs or {};params=params or {};parents=self._inputs(inputs)
        allowed={'object','target','preferred_xy','quaternion_wxyz','candidate_id','future_positions'}
        if not isinstance(params,dict) or set(params)-allowed:raise ValueError('Unsupported skill parameters')
        matches=[x for x in self.r.library if x['skill']==skill and x.get('object')==params.get('object') and
            (skill!='deposit' or x.get('target')==str(params.get('target')))]
        if len(matches)!=1:return dict(status='REJECTED',permission=False,
            reason='No qualified parameterized skill for this binding',required_params=['object']+(['target'] if skill=='deposit' else []),
            available=[dict(skill=x['skill'],object=x.get('object'),target=x.get('target')) for x in self.r.library if x['skill']==skill])
        row=copy.deepcopy(matches[0]);row['id']='open-'+uuid.uuid4().hex
        if 'future_positions' in params:
            count=params['future_positions']
            if type(count) is not int or not 1<=count<=12:raise ValueError('future_positions must be an integer from 1 to 12')
            row['future_positions']=count
        from .causal_plan import signature
        if all(self.r.c.truth(k)=='TRUE' for k in row['effects']):return dict(status='ALREADY_SATISFIED',effects=row['effects'],permission=False)
        if 'preferred_xy' in params:
            p=np.asarray(params['preferred_xy'],float)
            if p.shape!=(2,) or not np.isfinite(p).all():raise ValueError('Finite preferred XY required')
            row['preferred_xy']=p.tolist()
        if 'quaternion_wxyz' in params:
            q=np.asarray(params['quaternion_wxyz'],float)
            if q.shape!=(4,) or abs(np.linalg.norm(q)-1)>.001 or not np.allclose(Rotation.from_quat(np.roll(q,-1)).as_matrix()[:,2],[0,0,-1],atol=1e-6):raise ValueError('Deposit capability supports upright yaw only')
            row['quaternion_wxyz']=q.tolist()
        planning=None
        if hasattr(self.s,'preplanner') and skill in ('acquire','deposit'):
            planning=self.preview_skill(skill,params)
            if planning['status'] not in ('KINEMATIC_CANDIDATE','CAPABILITY_CONTRACT_ONLY'):
                return dict(status='REJECTED',permission=False,planning=planning,reason='Preplanning failed; inspect diagnostics before another proposal')
        if params.get('candidate_id'):row['candidate_id']=params['candidate_id']
        row['repair_of']='parameterized:'+signature(row)
        if planning and skill=='acquire' and planning.get('selected',{}).get('candidate_id'):row['candidate_id']=planning['selected']['candidate_id']
        self.r.c.register(row)
        result=self._result('skill_plan',dict(status='READY_FOR_LIVE_VERIFICATION',intent=row['id'],skill=skill,
            params=params,planning=planning,requires=row.get('requires',[]),requires_status={k:self.r.c.truth(k) for k in row.get('requires',[])},permission=False),parents)
        if hasattr(self.r,'register_candidate'):self.r.register_candidate(row,result['id'])
        self.plans[result['id']]=dict(kind='skill',row=row,inputs=inputs,used=False,
            motion_version=self.s.motion_version,held=self.s.held['object'] if self.s.held else None)
        return result

    def plan_motion(self,position,quaternion_wxyz,inputs=None,purpose='observation'):
        from .open_motion import interpolate_targets,check_tool_segment,empty_motion_scope
        inputs=inputs or {};parents=self._inputs(inputs)
        if self.r.c.quarantined or self.r.c.locks:raise ValueError('Physical effect unknown or robot resource busy')
        self._capture();s=self.s
        scope=None
        if s.held:s.current_held()
        else:
            scope=empty_motion_scope(s)
            if not scope['may_propose']:return dict(status='REJECTED',diagnostic=scope,permission=False)
        steps=interpolate_targets(s.scene.obs['robot_cartesian_pos'],position,quaternion_wxyz)
        if hasattr(s,'shared_trajectory'):
            planned=s.shared_trajectory.plan(position,quaternion_wxyz)
            self.store.append('TrajectoryPlanningResult',planned)
            if planned['admission']['status']!='PASS':
                check=planned['admission'];collision=planned['self_collision']
                return dict(status='REJECTED',permission=False,diagnostic={k:check.get(k) for k in ('status','reason','scope','attachment_included')},
                    last_sweep=check.get('checks',[None])[-1],kinematics=planned['validation'],
                    self_collision={k:collision.get(k) for k in ('status','minimum_signed_distance_m','samples','scope')},
                    detail_reference='harness/TrajectoryPlanningResult and trace shared_trajectory_plan')
            result=self._result('motion_plan',dict(status='READY_FOR_LIVE_VERIFICATION',target=position,
                quaternion_wxyz=quaternion_wxyz,source=s.scene.source,motion_version=s.motion_version,
                backend='existing_pyroki_trajopt',permission=False),parents)
            self.plans[result['id']]=dict(kind='trajectory',trajectory=planned,inputs=inputs,used=False,
                target=plain(position),q=plain(quaternion_wxyz),held=s.held['object'] if s.held else None,
                empty_scope=scope,motion_version=s.motion_version)
            return result
        check=check_tool_segment(s,position,quaternion_wxyz)
        key=digest([np.round(position,3).tolist(),np.round(quaternion_wxyz,3).tolist(),purpose])
        self.search_results.append(dict(strategy='motion',parameters=[np.round(position,3).tolist(),np.round(quaternion_wxyz,3).tolist()],outcome=check['status']))
        self.strategies[key]=self.strategies.get(key,0)+1
        if self.strategies[key]>2:return dict(status='REJECTED',reason='Equivalent motion strategy exhausted; acquire new information')
        if check['status']!='PASS':return dict(status='REJECTED',diagnostic=check,permission=False)
        if hasattr(s.reachable,'query'):
            ik=s.reachable.query(position,quaternion_wxyz)
            if ik['status']!='KINEMATIC_SOLUTION':
                return dict(status='REJECTED',reason='NO_SOLUTION_FOUND_not_proof_of_physical_unreachability',permission=False,
                    ik={k:ik.get(k) for k in ('id','status','failure_type','position_error_m','orientation_error_deg','cache_hit')})
        elif not s.reachable(position,quaternion_wxyz):return dict(status='REJECTED',reason='unreachable',permission=False)
        result=self._result('motion_plan',dict(status='READY_FOR_LIVE_VERIFICATION',target=position,
            quaternion_wxyz=quaternion_wxyz,purpose=purpose,checks=check,source=s.scene.source,
            motion_version=s.motion_version,permission=False),parents)
        self.plans[result['id']]=dict(kind='motion',target=plain(position),q=plain(quaternion_wxyz),steps=steps,
            motion_version=s.motion_version,used=False,inputs=inputs,
            start={k:plain(s.scene.obs[k]) for k in ('robot_cartesian_pos','robot_joint_pos','robot_gripper_joint_pos')},
            held=s.held['object'] if s.held else None,empty_scope=scope)
        return result

    def run_plan(self,plan_id):
        from .open_motion import check_tool_segment
        from .skill_contracts import SkillBoundaryStop
        plan=self.plans.get(plan_id)
        if not plan or plan['used']:raise ValueError('Unknown or already dispatched plan')
        self._inputs(plan['inputs'])
        if plan['kind']=='trajectory':
            if self.r.c.quarantined or self.r.c.locks:raise ValueError('Physical effect unknown or robot busy')
            s=self.s
            if plan.get('motion_version',s.motion_version)!=s.motion_version:raise ValueError('Trajectory plan stale')
            plan['used']=True;self.r.c.locks['open_motion']=plan_id
            receipt=getattr(getattr(getattr(self,'monitor',None),'boundary',None),'dispatch_receipt',None)
            before=receipt() if receipt else None;old_epoch=s.motion_version;returned=False;invalidated=False
            try:
                value=s.motion('follow_trajectory',plan['trajectory']);returned=True
                s.motion_version=old_epoch+1;self.motion_changed();invalidated=True;s.scene.frame()
                now=np.asarray(s.scene.obs['robot_cartesian_pos']);p=np.asarray(plan['target']);q=np.asarray(plan['q'])
                angle=np.degrees((Rotation.from_quat(np.roll(now[3:7],-1)).inv()*Rotation.from_quat(np.roll(q,-1))).magnitude())
                passed=np.linalg.norm(now[:3]-p)<.004 and angle<3
                if hasattr(s,'effect_monitor'):s.effect_monitor(p,q,passed)
                if not passed:raise SkillBoundaryStop('Planned trajectory TCP failed convergence',{})
                if s.held:
                    s.current_held()
                    self.reconcile_validation({'held:'+s.held['object']:dict(truth='TRUE',evidence={'source':s.scene.source,'reason':'current_hold_after_trajectory'})})
                else:self._reconcile_empty_motion(plan,plan_id)
                return self._result('motion_result',dict(status='PASS',verification=value,observation=self.observation()),[(plan_id,'attestation')])
            except BaseException as exc:
                if isinstance(exc,SkillBoundaryStop):self.reconcile_validation(exc.effects,str(exc))
                if s.held and (returned or receipt and receipt()['started']>before['started']):
                    self.reconcile_validation({'held:'+s.held['object']:dict(truth='UNKNOWN',evidence={'reason':'partial_trajectory_effect'})})
                self.r.c.graph.invalidate([plan_id],reason='trajectory interrupted; executed prefix must not be replayed')
                raise
            finally:
                dispatched=returned or receipt and receipt()['started']>before['started']
                if dispatched:
                    s.motion_version=old_epoch+1
                    if not invalidated:self.motion_changed()
                    self.r.planner.motion_epoch(s.motion_version)
                self.store.append('MotionTransaction',dict(plan=plan_id,dispatched=bool(dispatched),before=before,
                    after=receipt() if receipt else None,motion_version=s.motion_version,replay_prefix=False))
                self.r.c.locks.pop('open_motion',None)
        if plan['kind']=='skill':
            if hasattr(self.s,'preplanner') and (plan['motion_version']!=self.s.motion_version or plan['held']!=(self.s.held['object'] if self.s.held else None)):
                raise ValueError('Skill plan stale after robot or attachment change; preplan again')
            plan['used']=True
            value=self.r.execute(plan['row']['id']);return self._result('skill_result',value,[plan_id])
        s=self.s;c=self.r.c
        if c.quarantined or c.locks:raise ValueError('Physical effect unknown or robot resource busy')
        if plan['motion_version']!=s.motion_version:raise ValueError('Motion plan stale after physical change')
        s.scene.frame()
        from .open_motion import check_plan_start
        check_plan_start(plan,s)
        if not s.held:
            from .open_motion import empty_motion_scope
            if not empty_motion_scope(s)['may_propose']:raise ValueError('Empty-motion prerequisites changed')
        plan['used']=True
        c.locks['open_motion']=plan_id
        held='held:'+s.held['object'] if s.held else None
        try:
            for p,q in plan['steps']:
                s.scene.frame()
                if s.held:s.current_held()
                check=check_tool_segment(s,p,q)
                if check['status']!='PASS':raise SkillBoundaryStop('Current swept volume is unknown or occupied',{})
                if held:c.observe(held,'UNKNOWN',{'reason':'motion_effect_pending'},parents=[plan_id])
                s.open_motion_active=True
                try:s.move(p,q)
                finally:s.open_motion_active=False
                if s.held:s.current_held()
            if held:c.observe(held,'TRUE',{'source':s.scene.source,'reason':'current_hold_after_open_motion'},parents=[plan_id])
            else:self._reconcile_empty_motion(plan,plan_id)
        except BaseException:
            if s.held:s.phase='closed_unverified';s.effects[held]='UNKNOWN'
            c.graph.invalidate([plan_id],reason='motion verification failed')
            raise
        finally:
            c.locks.pop('open_motion',None)
            self.r.planner.motion_epoch(s.motion_version)
        # Only robot-dependent results and held geometry expire; fixed fixture
        # evidence remains historical and is revalidated at each use.
        for node,row in self.results.items():
            if row['type'] in ('robot','grasp','motion_plan') or (row['type']=='object' and s.held and row.get('object')==s.held['object']):
                c.graph.invalidate([node],reason='robot or attachment pose changed')
        self.r.planner.motion_epoch(s.motion_version)
        return self._result('motion_result',dict(status='PASS',observation=self.observation()),[(plan_id,'attestation')])

    def _reconcile_empty_motion(self,plan,plan_id):
        s=self.s
        if not (plan.get('empty_scope') or {}).get('reconcile_stalled'):return
        from .vision_skills import jaw_separation
        from .skill_contracts import SkillBoundaryStop
        if s.held or jaw_separation(s.scene.obs['robot_gripper_joint_pos'])<.075:
            raise SkillBoundaryStop('Open-jaw evidence changed during empty relocation',{})
        if not getattr(s,'stalled',None):return
        stopped=s.stalled['object'];s.stalled=None;s.phase='empty';s.active_condition=None
        self.reconcile_validation({'approach_stalled:'+stopped:dict(truth='FALSE',evidence={'source':s.scene.source,'reason':'verified_open_jaw_relocation'}),
            'robot_ready':dict(truth='TRUE',evidence={'source':s.scene.source,'reason':'verified_empty_relocation'})})

    def observation(self,capture=False):
        if capture:self.s.scene.frame()
        scene=self.s.scene;obs=getattr(scene,'obs',{})
        robot={k:plain(obs[k]) for k in ('robot_cartesian_pos','robot_joint_pos','robot_gripper_joint_pos') if k in obs}
        if 'robot_cartesian_pos' in robot:robot['tcp']=robot.pop('robot_cartesian_pos')[:7]
        if 'wrist_ft' in obs:robot['wrist_ft']=plain(obs['wrist_ft'])
        cameras={k:{'shape':list(v['images']['depth'].shape),'intrinsics':plain(v['intrinsics']),
            'pose':plain(v['pose_mat']),'rgbd_sha256':digest([np.asarray(v['images']['depth'])[::16,::16].tolist(),np.asarray(v['images']['rgb'])[::16,::16].tolist()])}
            for k,v in obs.items() if isinstance(v,dict) and 'images' in v}
        scope={'supported':'verified_held_vertical_probe_only','max_step_m':.02,'lateral_motion':False,'raw_release':False}
        if self.open_runtime:scope=dict(supported='parent_checked_local_cartesian_path',lateral_motion=True,rotation=True,max_distance_m=.15,max_rotation_deg=30,raw_release=False)
        a=getattr(self.s,'acquisition',None)
        if a:scope['grasp_origin']=plain(a['p']);scope['max_target_z']=float(a['p'][2]+.20)
        result=dict(source=scene.source,robot=robot,cameras=cameras,motion_scope=scope,
            phase=self.s.phase,held=self.s.held['object'] if self.s.held else None)
        result['information_hash']=digest([{k:v for k,v in robot.items() if k!='wrist_ft'},{k:v['rgbd_sha256'] for k,v in cameras.items()}])
        if capture:
            result['new_information']=result['information_hash']!=self.last_observation
            self.store.append('ObservationCaptured',result);self.last_observation=result['information_hash']
        return result

    def capture_observation(self):return self.observation(True)

    def inspect_region(self,role):
        if role not in ('source','destination'):raise ValueError('Unknown region role')
        if self.r.refreshes>=self.r.limit:raise ValueError('Shared evidence budget exhausted')
        self.r.refreshes+=1
        self.s.scene.frame()
        try:
            if not self.s.anchors:self.s.prepare(capture=False)
            result=dict(status='PASS',geometry=plain(self.s.scene.observe_region(role)),source=self.s.scene.source)
            self.information[role]={'coverage':1.};return result
        except ValueError as exc:
            diagnostic=plain(getattr(self.s.scene,'region_diagnostics',{}).get(role,{}))
            self.information[role]=diagnostic
            return dict(status='UNKNOWN',reason=str(exc),diagnostic=diagnostic,source=self.s.scene.source)

    def verify_conditions(self,keys):
        if not isinstance(keys,list) or len(keys)>32 or any(k not in self.r.c.definition.get('condition_specs',{}) for k in keys):raise ValueError('Unknown condition query')
        result=self.r._planned_refresh(keys)
        self.store.append('VerificationResult',dict(scope=keys,observed=result,task_success_claim=False))
        return result

    def move_tcp(self,position,quaternion_wxyz,operation_id):
        if not isinstance(operation_id,str) or not 1<=len(operation_id)<=100:raise ValueError('Operation ID required')
        if operation_id in self.operations:raise ValueError('Operation already attempted; no physical replay')
        if self.open_runtime:
            self.operations.add(operation_id);plan=self.plan_motion(position,quaternion_wxyz)
            return self.run_plan(plan['id']) if 'id' in plan else plan
        s=self.s;s.scene.frame()
        if s.phase!='held_verified' or s.held is None:raise ValueError('Verified held state required')
        s.current_held()
        a=s.acquisition
        p,q=admitted_observation_target(s.scene.obs['robot_cartesian_pos'],a['p'],position,quaternion_wxyz)
        # Existing recovery verifier: current visible neighbor/receptacle sweep,
        # robot orientation, source corridor and kinematic reachability.
        s._corridor(s.held['object'],probe=True)
        if not s.reachable(p,q):raise ValueError('Observation target unreachable')
        self.operations.add(operation_id)
        before=self.observation()
        # A newly composed operation has a versioned graph node. Old tool plans
        # are invalid once dispatch may change the view; verified sibling goals
        # remain valid until contradictory evidence arrives.
        c=self.r.c
        node=c.graph.add('api_call',dict(name='move_tcp',operation_id=operation_id,target=plain(p),status='in_flight'),
            parents=[c.program_node] if c.program_node else [])
        for key,old in list(c.conditions.items()):
            if c.definition.get('condition_specs',{}).get(key,{}).get('kind')=='tool_data':
                c.graph.invalidate([old],reason='open observation motion changed evidence applicability')
        held_key='held:'+s.held['object']
        c.observe(held_key,'UNKNOWN',dict(reason='observation_motion_requires_fresh_hold_validation'),parents=[node])
        try:s.move(p,q);s.scene.frame();s.current_held()
        except BaseException:
            c.graph.nodes[node]['data']['status']='failed'
            c.graph.invalidate([node],reason='public observation motion failed verification')
            s.phase='closed_unverified';s.effects[held_key]='UNKNOWN'
            raise
        c.graph.nodes[node]['data']['status']='done'
        c.graph.nodes[node]['state']='done';c.graph._event('done',node_id=node)
        c.observe('held:'+s.held['object'],'TRUE',dict(source=s.scene.source,reason='current_hold_after_guarded_observation_motion'),parents=[node])
        self.r.planner.motion_epoch(s.motion_version)
        after=self.observation()
        self.store.append('VerificationResult',dict(operation_id=operation_id,before=before,after=after,held_verified=True,task_success_claim=False))
        return dict(status='PASS',operation_id=operation_id,observation=after)

    def graph_query(self,query,key=None):
        c=self.r.c
        if query=='state':return self.r.state()
        if query=='failures':return copy.deepcopy(self.r.events[-12:])
        if key not in c.conditions:raise ValueError('Unknown evidence key')
        node=c.conditions[key]
        if query=='evidence':return copy.deepcopy(c.graph.nodes[node])
        if query=='dependencies':return c.graph.diagnose([node],max_nodes=16)
        raise ValueError('Unknown graph query')

    def functions(self):
        names=['capture_observation','inspect_region','verify_conditions','move_tcp','graph_query']
        if self.open_runtime:names+=['read_robot','observe_object','observe_grasp','observe_surfaces','plan_skill','plan_motion','run_plan','recall_result','belief_query','history_query','observation_opportunities']
        if hasattr(self.s,'preplanner'):names+=['preview_skill','planning_query','decision_state']
        if hasattr(self,'capabilities'):names+=['capability_tools','register_capability','run_capability','capability_query']
        return {k:getattr(self,k) for k in names}

V35_API_DOC = """
V35 planning additions (same tools for both context encodings):
preview_skill(skill, params={}): NO physical motion; actual candidate/IK search.
Returns KINEMATIC_CANDIDATE, NO_SOLUTION_FOUND, EVIDENCE_OR_CAPABILITY_MISSING,
or CAPABILITY_CONTRACT_ONLY for staged recovery. No status is a motion permission.
plan_skill now runs preplanning for acquire/deposit before creating executable ID.
Planning may capture current public evidence but never opens/closes/moves the robot.
planning_query(id=None): list searches; query search or ik ID for full candidates,
position/rotation residuals, solver iterations, references and joint limits.
NO_SOLUTION_FOUND is NOT physical unreachability. Only a bounded set was tested.
Deposit search compares upright yaw orientations with the original geometric margin;
endpoint feasibility does NOT certify transit or final release. run_plan retains all
live checks and final capture. Cache hits are not new information or task progress.
decision_state(): compact goal-oriented public state and search summary. attachment
is string object_name or None, SAME TYPE as execution_state().attachment.
attachment_estimate is a separate dict or None with T_tcp_shape_frame,
object_center_in_tcp, current_status, source and model-validity requirements.
Use diagnostics to compare meaningful alternatives before proposing physical actions.
Full six-DOF grasp execution is unavailable without an applicable swept-volume
contract; raw GraspNet proposals remain diagnostic evidence, never motion authority.
"""


def runtime_api_doc(*,preplanning=False,semantic=False):
    """One tool contract for initial, recovery and semantic proposal contexts."""
    result=OPEN_API_DOC+(V35_API_DOC if preplanning else '')
    if semantic:
        from .semantic_capabilities import SEMANTIC_API_DOC
        result+=SEMANTIC_API_DOC
    return result
