"""Public Python recovery tools; all physical dispatch stays in the parent.

This version exposes the existing certified vertical probe corridor, not an
unqualified arbitrary Cartesian planner. No raw release or simulator handles.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .agent_events import plain,digest


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
    def __init__(self,runtime,store):self.r=runtime;self.s=runtime.skills;self.store=store;self.operations=set();self.last_observation=None

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
        try:return dict(status='PASS',geometry=plain(self.s.scene.observe_region(role)),source=self.s.scene.source)
        except ValueError as exc:return dict(status='UNKNOWN',reason=str(exc),source=self.s.scene.source)

    def verify_conditions(self,keys):
        if not isinstance(keys,list) or len(keys)>32 or any(k not in self.r.c.definition.get('condition_specs',{}) for k in keys):raise ValueError('Unknown condition query')
        result=self.r._planned_refresh(keys)
        self.store.append('VerificationResult',dict(scope=keys,observed=result,task_success_claim=False))
        return result

    def move_tcp(self,position,quaternion_wxyz,operation_id):
        if not isinstance(operation_id,str) or not 1<=len(operation_id)<=100:raise ValueError('Operation ID required')
        if operation_id in self.operations:raise ValueError('Operation already attempted; no physical replay')
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

    def functions(self):return {k:getattr(self,k) for k in ('capture_observation','inspect_region','verify_conditions','move_tcp','graph_query')}
