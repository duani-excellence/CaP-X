"""Task-aware shared execution substrate and selectable graph recovery.

Extends the existing parent broker/graph, not an Agent framework. All online
inputs are filtered public observations. Physical evaluators stay in the runner.
"""
from __future__ import annotations
import copy
import time
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .graph_feedback import GraphFeedbackMonitor, PreMotionRejected
from .bounded_ik import IKConvergenceError
from .feedback import FeedbackMonitor, public_observation
from .sandbox import ExecutionInterrupted
from .public_geometry import capture_geometry, object_geometry, measured_surface, place_metrics, choose_table_region
from .grounded_targets import compile_target
from .grasp_geometry import cgn_upright_panda, panda_open_sweep
from .grounded_control import precise_goto
from .support_geometry import color_points, current_support, reliable_top, retained_consistent
from .grip_offset import calibrate as calibrate_attachment, predict as predict_attachment, check_surface as check_attachment

APIS=['observe_object','observe_grasp','observe_surfaces','observe_table','reobserve_table','read_robot','plan_skill','run_plan','recall_result','verify_hold','verify_task','declare_intent','prepare_evidence']
API_DOC='''Grounded Vision API, robot-base metres and WXYZ. Actual SAM3/GraspNet and calibrated RGB-D are used. No simulator truth.
Sandbox: use self-contained Python with only numpy imports. Names beginning with underscore (except the single '_' loop placeholder) and private attributes are forbidden; use id_of, not _id.
Perception failure interrupts the entire actor; observe_* does NOT return None on failure. Do not call a known-failing observation and then branch on None or try to catch the interruption. Use recovery_preconditions before retrying. If a held cube occludes its support at placement height, plan a bounded upward observation move from a fresh robot source before retrying support observation; existing phase/control/budget checks still apply. Never release without current permission.
IMPORTANT: every observe_* and read_robot returns a dictionary with an 'id' string and data fields, NOT a string. Extract result['id'] for plan_skill inputs. plan_skill itself returns a plan ID string. Example: grasp=observe_grasp('red cube'); p=plan_skill('approach', {'grasp':grasp['id']}); run_plan(p). Example: robot=read_robot(); p=plan_skill('lift', {'robot':robot['id']}); run_plan(p). Never pass a whole source dictionary as a dependency.
observe_object(name): SAM3 position plus public RGB-D geometry. May observe a nonheld object while holding another; this does not move or release the robot. Held-object size is never recalibrated from occluded held observations. Returns dictionary with id and public fields. Names: red cube, green cube (no green in Lift).
observe_grasp(name): real GraspNet candidate fused with visible cube centre and an upright yaw-preserving grasp. Observe object first. Returns dictionary with id and public fields.
observe_surfaces(name): current RGB-D surface with known upright height. If a held object is partially hidden, the parent may make at most two bounded wrist-yaw observations at verified public clearance; robot/held-source IDs then expire. Read robot AFTER surfaces. Returns dictionary with id and public fields; rigid attachment remains a conditional assumption, not contact proof.
observe_table(name): before table placement, observe_object for every nonheld task object to establish public geometry needed when the held object later occludes the patch. Find a currently visible clear table patch for temporarily placing that object (Restack green). While holding, candidate search uses fresh validated held geometry; if unavailable, observe_surfaces for the held object before retrying or stop. Search bounds are historical intent, not placement/release permission. All current table checks remain required. A table target rejected by IK before motion or blocked by unresolved elevated geometry is excluded from this grasp's future table searches; observe_table may return a different current region. The exclusion is historical search memory, not a statement that the area is occupied or globally unreachable. Returns dictionary with id and public fields, or interrupts when unknown.
reobserve_table(region_id, change_view=True): revalidate the SAME prior table bounds after uncertain clearance; region_id may be invalid and serves only as historical position intent. With change_view=True the parent attempts at most one bounded wrist-yaw at verified public clearance, sharing the configured episode view limit and local-action budget. Each grasp/location context has only +20 and -20 degree candidates once each; capture, new actors and minor drift never replenish them. A returned non-probe translation of at least 0.03 m permits a new context; returning to a previously used location reuses its candidates. With False only fresh perception is attempted. Only a current PASS returns a NEW region id; old plans remain invalid. Then refresh held surface and robot, make a new table_place plan. Failure interrupts with remaining budgets; never blindly descend or release.
read_robot(): current encoder/TCP source, returns dictionary with id and pose. No object truth.
plan_skill(skill, inputs, params=None): returns a plan ID. inputs maps role names to PARENT IDs, never numeric targets. Server computes the target; arbitrary XYZ/quaternions are forbidden.
  approach: inputs={'grasp': grasp_id}; params={'z_approach':0.1} optional.
  lift: inputs={'robot': fresh_robot_id}; params={'dz':0.12} optional, <=0.25. At most one lift while holding is unverified.
  place: inputs={'robot':fresh_robot_id,'held_surface':fresh_held_surface_id,'support_surface':fresh_support_surface_id}; params={'gap':0.006} optional.
  table_place: inputs={'robot':fresh_robot_id,'held_surface':fresh_held_surface_id,'table_region':table_id}; params={'gap':0.006} optional.
  retreat: inputs={'robot':fresh_robot_id}; params={'dz':0.10} optional, >=0.08. Vertical first retreat after release.
run_plan(id): consume plan once, validate current versions and phase, execute and verify at API boundaries. IMPORTANT: place and table_place perform motion ONLY; they NEVER open the gripper or release. After run_plan(place_id) returns, explicitly call open_gripper(); then r=read_robot(); run_plan(plan_skill('retreat', {'robot':r['id']})). A retreat while still holding is rejected. Plan just before execution. Motion invalidates old robot/held-object pose and downstream plans. Local release alignment also invalidates old retreat.
open_gripper(): initial opening, or verified place/table_place release. After release the NEXT physical action must be a fresh retreat. Do not open twice, close, or home before retreat.
close_gripper(): only after a returned approach with an empty gripper. Repeating close while already holding is forbidden and does not reset the grasp stage.
verify_hold(): bounded public visual co-motion probe if lift baseline was unavailable. If the held object is occluded, it first attempts one already-budgeted yaw view using the same public attachment, height and neighbor-clearance constraints; only the subsequent co-motion check can verify hold. Does not lift repeatedly. May interrupt as unrecoverable when surfaces remain hidden or probe budget expires.
prepare_evidence(): acquire an identifiable held-geometry anchor for forthcoming placement/release. Only while already holding_verified. Uses current geometry or at most one existing clearance-checked wrist view; it cannot grant release, renew an old observation, or restore lost grasp. Then acquire fresh source handles and create a new plan. An equivalent view or new source ID is not progress.
recall_result(id): use a valid retained source without service calls; invalid sources are rejected. New actor variables vanish; physical stage, completed actions and episode budgets persist.
home_pose(): only empty, after verified retreat. Never home while holding.
Stack: grasp red, lift, place on green, release, retreat.
Lift: grasp red, lift with evidence, keep holding; DO NOT release.
Restack: green begins on red. First move green to a verified empty table patch, release and retreat. Then red onto green, release and retreat.
Recovery: inspect phase, grasp_active, recovery_scope and recovery_preconditions. If grasp_active is false and phase is ready, no object is currently held; last_manipulated_object and last_grasp_loss are history only. Open empty jaws, observe the actual scene, and replan remaining work. Do not place or retreat as though a historical grasp still existed. On pre_motion_table_target_rejected or table_clearance_current_nonheld_obstacle_in_table_region, call observe_table again to obtain an alternative, then refresh held surface and robot and create a new table_place plan. Do not repeatedly reobserve or plan the rejected region; never repeat grasp/lift just to change the table target. For missing_geometry_objects, call observe_object on each named nonheld object, then reobserve_table(region_id, change_view=False), refresh held/robot sources and create a new plan; no wrist movement is needed just to acquire missing geometry. If a returned place was followed by retreat_requires_release, release is still pending: revalidate as required, explicitly open_gripper(), then fresh retreat. If holding_unverified call verify_hold; do not close again or repeatedly raise. If released, fresh retreat only. If holding_verified reobserve only required invalid sources, compute fresh place target. Do not replay completed physical actions. Missing evidence can require an explicit unsuccessful stop.
'''

V22_API_DOC=API_DOC+'''\nverify_task(): refresh deployment-observable task conditions. Does not move, release, or return physical reward. Restack green-on-table is an INTERMEDIATE goal; red-on-green and safe retreat are still required. UNKNOWN requires evidence, not replay of historical actions. Returned API calls are not proof of their expected effects.\n'''


V23_API_DOC=V22_API_DOC+"""
V23 goal contract: the parent registers the task goals, prerequisite conditions and future skill intents BEFORE executing the program. They are intent, never current motion/release permission. declare_intent(rows) may additionally declare the program's ordered intended skills before the first physical call; rows contain skill, object, destination, and optional after (indices of earlier rows). Only bindings present in the parent task contract are legal. plan_skill/run_plan are linked to those intents and still undergo every parent boundary check. Python code remains a complete program; no line-by-line resumption.
observe_surfaces may return evidence_kind='predicted', current_visual_detection=False and an error_bounds dictionary only when historical identifiable geometry, public robot propagation, calibration, expected RGB-D occlusion and the current safety contract all permit it. It is never a fresh visual detection. Missing/contradictory/expired or too-wide predictions interrupt; repeatedly asking the same occluded view does not improve evidence. verify_task checks complete declared goals, not historical return receipts.
"""


class GroundedMonitor(GraphFeedbackMonitor):
    def __init__(self,mode,task,observe,*,emit=None,graph_emit=None,max_local=3,max_episode_local=8,max_episode_views=2,fault='none',site=None,graph_options=None,motion_transform=None,ik_success_count=None,task_conditions=False,task_definition=None,memory_profile=None,evidence_planning=False):
        from .task_definition import resolve_definition
        self.task_definition=resolve_definition(task_definition if task_definition is not None else task)
        if type(max_episode_views) is not int or max_episode_views<0:raise ValueError('Invalid episode view budget')
        options=dict(diagnostic_slice=True,selective_invalidation=True,selective_recovery=True)
        if graph_options and (set(graph_options)-set(options) or any(type(v) is not bool for v in graph_options.values())):raise ValueError('Invalid graph options')
        options.update(graph_options or {})
        self.graph_options={k:False for k in options} if mode=='G0' else options
        self.motion_transform=motion_transform
        self.ik_success_count=ik_success_count
        self.table_evidence_fn=None;self.current_public_state=None
        self.color_points_fn=color_points
        self.geometry_fn=capture_geometry
        super().__init__(mode,observe,emit=emit,graph_emit=graph_emit,max_local=max_local,
                         fault='support_refresh' if fault=='support_refresh' else 'none',site=site)
        if fault not in ('none','support_refresh','missing_baseline','target_shift'):raise ValueError('Unknown fault')
        self.fault=fault;self.task=task;self.phase='ready';self.held_object=None;self.approached_object=None
        self.geometry={};self.geometry_ids={};self.baseline=None;self.close_tcp_z=None;self.table_z=None;self.points=np.empty((0,3))
        self.lift_attempts=0;self.hold_probes=0;self.local_attempts=0;self.max_episode_local=max_episode_local
        self.last_place=None;self.terminal_reason=None;self.object_epochs={o:0 for o in self.task_definition['objects']}
        self.recovery_outcomes=[];self.attachment=None;self.attachment_id=None
        self.support_history={};self.colored_points={};self.active_authorization=None
        self.active_dependencies=None
        self.release_permission=None
        self.skill_intent=None
        self.timing=dict(public_observation_seconds=0.,control_seconds=0.,perception_api_seconds=0.,monitor_dispatch_seconds=0.)
        self.contract_calls={}
        self.visibility_probes=0;self.visibility_reference_q=None;self.perception_refinements=0
        self.max_episode_views=max_episode_views
        self.visibility_contexts=[];self.visibility_context=None
        self.visibility_translation_m=.03  # Resource-policy separation, never a geometry acceptance tolerance.
        self.observation_calibration_sources=[];self.rejected_table_targets=[];self.last_grasp_loss=None
        self.memory=None;self.evidence_reference=None
        self.evidence_planning=evidence_planning;self.evidence_request=None
        if memory_profile:
            from .state_memory import StateMemory
            self.memory=StateMemory(self.graph,memory_profile)
        self.conditions=None
        if task_conditions:self.enable_task_conditions()

    def enable_task_conditions(self):
        if self.conditions is None:
            from .task_conditions import TaskConditions
            self.conditions=TaskConditions(self.graph,self.task_definition)
        return self.conditions

    def task_definition_destination(self,obj):
        return next((r['destination'] for r in self.task_definition['subgoals'] if r['object']==obj),None)

    def _object_binding(self,name):
        if name not in self.task_definition['objects']:raise ValueError('Unknown task object')
        return self.task_definition['objects'][name]

    def register_camera_calibration(self,metadata):
        """Bind every corrected frame to its exact public projection version.

        A changed projection invalidates its dependent observations/plans. An
        unchanged projection is reused unless conservative recovery invalidated
        it; it never imports object poses or an empirical scene correction.
        """
        sources=[]
        missing=[node for node in self.observation_calibration_sources
            if self.graph.nodes[node]['data']['camera'] not in metadata and self.graph.nodes[node]['valid']]
        if missing:
            self.graph.invalidate(missing,'camera_projection_unavailable',conservative=not self.graph_options['selective_invalidation'])
        for name,data in sorted(metadata.items()):
            key='camera_calibration:'+name
            payload=copy.deepcopy(data)
            fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
            old=self.graph.latest.get(key)
            if old and self.graph.nodes[old]['valid'] and self.graph.nodes[old]['data']['sha256']==fingerprint:
                source=old
            else:
                source=self.graph.version(key,'estimate',{'camera':name,'calibration':payload,'sha256':fingerprint},
                    reason='camera_projection_version_changed')
            sources.append(source)
        self.observation_calibration_sources=sources
        return list(sources)

    def capture(self):
        start=time.monotonic()
        try:return self._capture()
        finally:self.timing['public_observation_seconds']+=time.monotonic()-start

    def _capture(self):
        state=public_observation(self.observe())
        self.current_public_state=state
        self.last_public={k:np.asarray(v).tolist() for k,v in state.items() if not isinstance(v,dict)}
        self.last_evidence,self.table_z,self.points=self.geometry_fn(state)
        self.colored_points={name:self.color_points_fn(state,self._object_binding(name)['perception_name']) for name in self.object_epochs}
        # Retire prior-world evidence before registering this newly acquired
        # frame. In G0 a real support update discards all old work, including
        # calibration estimate nodes; it must not invalidate its own new frame.
        calibration={self.graph.nodes[i]['data']['camera']:copy.deepcopy(self.graph.nodes[i]['data']['calibration'])
                     for i in self.observation_calibration_sources}
        current_surfaces=[]
        for name,geom in self.geometry.items():
            surface=self._surface(name)
            if name==self.held_object and self.closed or not reliable_top(surface,geom):continue
            prior=self.support_history.get(name)
            if prior and (np.linalg.norm(np.asarray(surface['center'])[:2]-np.asarray(prior['surface']['center'])[:2])>.008 or abs(float(surface['hi'][2])-float(prior['surface']['hi'][2]))>.006):
                changed=[n['id'] for n in self.graph.nodes.values() if n['kind']=='estimate' and n['valid'] and (n['data'] or {}).get('object')==name]
                self.graph.invalidate(changed,'public_support_pose_changed',conservative=not self.graph_options['selective_invalidation'])
                self.object_epochs[name]+=1
            current_surfaces.append((name,surface))
        if any(not self.graph.nodes[i]['valid'] for i in self.observation_calibration_sources):
            # Re-register only this capture's already obtained public projection;
            # no private state or additional perception request is introduced.
            self.register_camera_calibration(calibration)
        self.last_observation=self.graph.add('observation',{'robot':self.last_public,'surfaces':self.last_evidence,
            'table_z':self.table_z,'source':'public_RGBD_and_robot_proprioception',
            'evidence_ref':self.evidence_reference},parents=self.observation_calibration_sources)
        for name,surface in current_surfaces:
            fact=self.graph.add('verification_fact',{'type':'historical_full_support_surface','object':name,'object_epoch':self.object_epochs[name],'surface':surface},parents=[self.last_observation,self.geometry_ids[name]])
            self.support_history[name]={'epoch':self.object_epochs[name],'surface':copy.deepcopy(surface),'source_id':fact}
        if self.memory:self._refresh_memory()
        if self.conditions:self.conditions.refresh(self)
        return state,self.last_evidence

    def record(self,layer,check,status,reason,metrics=None,action='continue',graph_parents=(),**extra):
        row=FeedbackMonitor.record(self,layer,check,status,reason,metrics,action,**extra)
        if hasattr(self,'graph'):
            parents=[p for p in (self.last_call,self.last_observation) if p]+list(graph_parents)
            if layer=='L2' and check in ('grasp','lift','hold_probe'):
                if self.baseline and self.baseline.get('source_id'):parents.append(self.baseline['source_id'])
                if check!='grasp' and self.skill_facts.get('grasp'):parents.append(self.skill_facts['grasp'])
            self.last_check=self.graph.add('check',{'layer':layer,'check':check,'status':status,'reason':reason,'metrics':metrics or {}},parents=parents)
            if layer=='L2' and check in ('grasp','lift','hold_probe','pre_release'):
                # Facts retain what was observed then, never current permission.
                if check=='grasp':
                    self.skill_facts.clear()
                self.skill_facts[check]=self.graph.add('verification_fact',{
                    'historical_only':True,'check':check,'status':status,
                    'source_check_id':self.last_check,'grasp_epoch':self.grasp_epoch,
                    'object':self.held_object,'metrics':metrics or {}},parents=[self.last_check])
                if self.conditions:self.conditions.check_event(self,check,status,self.skill_facts[check])
            if status!='PASS':self.roots=[self.last_check]+([self.active_plan] if self.active_plan else [])
        return row


    def _physical_effect_unknown(self,name,node):
        # Keep the previous phase as history, never as current permission. This
        # API has no controller completion receipt/cancellation protocol; RGB-D
        # and finger encoders alone cannot certify the unobserved path effects.
        previous=self.phase
        self.phase='reconciliation_required'
        self.terminal_reason='unknown_physical_effect_terminal_stop'
        super()._physical_effect_unknown(name,node)
        self.reconciliation_required['previous_phase']=previous
        self.attachment=None;self.attachment_id=None;self.last_place=None
        if self.memory:
            for obj in self.object_epochs:self.memory.invalidate(obj,'unknown_effect')
        if self.conditions:
            for key in list(self.conditions.ids):
                self.conditions.set(key,'UNKNOWN',reason='unknown_physical_effect_quarantine')
        self.record('L1','action_completion','UNCERTAIN','unknown_physical_effect',
            {'api':name,'previous_phase':previous,'public_robot':self.last_public,
             'controller_completion_receipt':'unavailable','recoverable':False},action='terminal_stop')

    def invoke(self,name,args,kwargs,fn):
        # The trusted boundary authorizes only this invocation and its bounded
        # local continuations. No old PASS can authorize a later actor/API.
        self.active_authorization=None;self.active_precondition=None
        start=time.monotonic();before=dict(self.timing)
        self.contract_calls[name]=self.contract_calls.get(name,0)+1
        try:return super().invoke(name,args,kwargs,fn)
        except ExecutionInterrupted:
            if self.mode=='G1' and not self.graph_options['selective_recovery']:
                self.graph.invalidate(self.roots,'ablation_conservative_recovery_suffix',conservative=True)
            raise
        finally:
            self.active_authorization=None;self.active_precondition=None;self.release_permission=None
            nested=sum(self.timing[k]-before[k] for k in ('public_observation_seconds','control_seconds','perception_api_seconds'))
            self.timing['monitor_dispatch_seconds']+=max(0.,time.monotonic()-start-nested)

    def _require_current_grasp(self,operation):
        # Public finger encoders can refute an existing grasp before a command.
        # A nonempty aperture alone cannot verify it: the existing geometry and
        # co-motion checks remain necessary. Never infer the object's location.
        if not self.closed or self.phase not in ('holding_unverified','holding_verified'):
            return
        state=np.asarray(self.last_public.get('robot_cartesian_pos',[]))
        if state.ndim!=1 or len(state)<8 or not np.isfinite(state[7]):
            self.record('L1','current_gripper_state','UNCERTAIN',
                'current_gripper_state_required_before_action',{'api':operation},action='reobserve')
            self._stop('current_gripper_state_required_before_action',metrics={
                'recoverable':True,'physical_action_issued':False,
                'next_action':'refresh_public_robot_state_before_retry'})
        if float(state[7])<.05:
            self.graph._event('public_grasp_pre_action_rejection',api=operation,
                observation_id=self.last_observation,physical_action_issued=False)
            self._grasp_lost('empty_jaws_before_action')

    def _phase_guard(self,operation,allowed,reason,roots=()):
        self.capture()
        self._require_current_grasp(operation)
        parents=list(roots)+list(self.skill_facts.values())
        if self.conditions:
            obj=self.held_object or self.approached_object
            skill='open_gripper' if operation=='release' else operation
            intent=self.conditions.intent(skill,obj)
            if intent:parents.append((intent,'attestation'))
        self.record('L2','phase_precondition','PASS' if allowed else 'FAIL',reason,
            {'operation':operation,'phase':self.phase,'closed':self.closed,
             'lifted':self.lifted,'held_object':self.held_object,
             'lift_attempts':self.lift_attempts,'grasp_epoch':self.grasp_epoch,
             'historical_facts_are_not_permission':True},
            action='continue' if allowed else 'replan',graph_parents=parents)
        gate=self.last_check
        if not allowed:self._stop(reason,list(roots)+[gate])
        self.active_authorization=self.graph.add('verification_fact',{
            'historical_only':True,'type':'boundary_authorization',
            'operation':operation,'source_check_id':gate,'block':self.block},parents=[gate])
        return gate

    def _tracked(self,fn):
        def physical(name,args,kwargs):
            start=time.monotonic()
            try:return physical_impl(name,args,kwargs)
            finally:
                key='control_seconds' if name in ('goto_pose','open_gripper','close_gripper','home_pose') else 'perception_api_seconds'
                self.timing[key]+=time.monotonic()-start
        def physical_impl(name,args,kwargs):
            # A trusted counter is sampled at the complete API boundary. Equal
            # successful-solve counts prove even a multi-segment goto rejected
            # its first solver before any segment could move. A later segment's
            # rejection must keep the unknown-effect quarantine.
            before=None
            if name=='goto_pose' and self.ik_success_count is not None:
                value=self.ik_success_count()
                if type(value) is int and value>=0:before=value
            try:
                if name=='goto_pose' and self.motion_transform is not None:
                    p,q=self.motion_transform(args[0],args[1])
                    return fn(name,[p,q,*args[2:]],kwargs)
                return fn(name,args,kwargs)
            except (IKConvergenceError,ConnectionError) as outer:
                # The existing audited API wraps non-value exceptions. Only its
                # typed cause can carry solver evidence; text is never proof.
                exc=outer if isinstance(outer,IKConvergenceError) else outer.__cause__
                if not isinstance(exc,IKConvergenceError):raise
                # In the existing API, z_approach != 0 executes two segments.
                # The first may already have moved before the second rejects.
                single_segment=(name=='goto_pose' and float(kwargs.get('z_approach',args[2] if len(args)>2 else 0.))==0.)
                no_returned_segment=(before is not None and type(exc.successful_calls_before) is int
                                     and exc.successful_calls_before==before)
                if name=='goto_pose' and (single_segment or no_returned_segment) and exc.checked_no_segment_execution:
                    raise PreMotionRejected(str(exc)) from outer
                raise
        tracked=super()._tracked(physical)
        def call(name,args,kwargs):
            if name not in ('goto_pose','open_gripper','close_gripper','home_pose'):
                return tracked(name,args,kwargs)
            if not self.active_authorization:
                self._stop('missing_current_phase_authorization',incomplete=True)
            # Motion may invalidate the entry check. Each bounded retry gets a
            # fresh check against current public state; the historical receipt
            # explains its authorized boundary without replaying that boundary.
            self.capture()
            self._require_current_grasp(name)
            self._check_active_dependencies(args if name=='goto_pose' else None)
            if name=='open_gripper' and self.closed:
                # Immutable authorization describes history. Current permission
                # must survive the LAST observation, then be verified afresh.
                gate=self.release_permission
                if not gate or not self.graph.nodes[gate]['valid']:
                    self._stop('release_permission_expired_after_final_observation',
                        [gate] if gate else [],metrics={'recoverable':True})
                check=self._release_evidence(self.last_place)
                if check['status']!='PASS':
                    self._stop('release_final_evidence_'+check['reason'],metrics={'recoverable':True})
                self._require_current_grasp(name)
                self.release_permission=self.last_check
                self._valid(self.release_permission)
            self.record('L2','phase_precondition','PASS','authorized_current_boundary',
                {'api':name,'phase':self.phase,'closed':self.closed,'lifted':self.lifted,
                 'held_object':self.held_object,'grasp_epoch':self.grasp_epoch,
                 'historical_facts_are_not_permission':True},
                graph_parents=[self.active_authorization]+list(self.skill_facts.values())+([self.release_permission] if name=='open_gripper' and self.closed else []))
            self.active_precondition=self.last_check
            try:
                if self.active_dependencies is not None:self.active_dependencies['started']=True
                return tracked(name,args,kwargs)
            finally:self.active_precondition=None
        return call

    def _check_active_dependencies(self,motion=None):
        """Revalidate a multi-call skill before every actual physical API boundary.

        Own returned motion expires robot/held estimates. It does not authorize a
        changed external support or a new source version to reuse the old target.
        """
        contract=self.active_dependencies
        if contract is None:return
        plan_id=self.active_plan
        changed=[];missing=[];conditional=[]
        for source in contract['sources']:
            node=self.graph.nodes[source];key=node.get('key')
            if key and self.graph.latest.get(key)!=source:changed.append(source)
            elif not contract['started'] and not node['valid']:changed.append(source)
        for name,expected in contract['external_epochs'].items():
            if self.object_epochs[name]!=expected:changed.extend(contract['external_sources'].get(name,[]))
        for name,reference in contract.get('external_references',{}).items():
            current=self._surface(name);geom=self.geometry.get(name)
            if not reliable_top(current,geom):
                current=current_support(self.colored_points.get(name,np.empty((0,3))),self.table_z,geom,camera_origin=self._current_camera_origin())
            if current is None and self.memory:
                current=self._predicted_surface(name)
                if current:conditional.append(name)
            if current is None:
                # A partial current view can only support a clear, non-descending
                # transit under explicit static-support assumptions. It is never
                # fresh localization or permission to descend/release.
                held=self._held_surface() if motion is not None else None
                tcp=np.asarray(self.last_public['robot_cartesian_pos'])
                target=np.asarray(motion[0]) if motion is not None else None
                clear=(held is not None and target is not None
                    and min(float(held['lo'][2]),float(held['lo'][2]+target[2]-tcp[2]))
                        >=float(reference['hi'][2])+.05
                    and target[2]>=tcp[2]-.002)
                allowed=(contract.get('stage') in ('ascend','above_target') and clear
                    and retained_consistent(self.colored_points.get(name,np.empty((0,3))),reference,geom))
                if allowed:conditional.append(name)
                else:missing.append(name)
            if current and (np.linalg.norm(np.asarray(current['center'])[:2]-np.asarray(reference['center'])[:2])>.008
                            or abs(float(current['hi'][2])-float(reference['hi'][2]))>.006):
                changed.extend(contract['external_sources'].get(name,[]))
        if changed:
            changed=list(dict.fromkeys(changed))
            self.graph.invalidate(changed,'active_skill_external_dependency_changed',conservative=not self.graph_options['selective_invalidation'])
            self.record('L2','active_dependencies','FAIL','active_skill_external_dependency_changed',
                {'plan_id':plan_id,'changed_sources':changed,'already_started':contract['started'],
                 'completed_calls':list(self.call_history)},action='replan',graph_parents=changed)
            self._stop('active_skill_external_dependency_changed',[plan_id,self.last_check]+changed)
        if self.conditions:
            required=self.graph.nodes[plan_id]['data'].get('required_conditions',[])
            failed=[key for key in required if self.conditions.get(key)['truth']=='FALSE']
            unknown=[key for key in required if self.conditions.get(key)['truth']=='UNKNOWN']
            # A missing Restack stage witness can only inherit the SAME explicit
            # clear-transit policy as its support geometry. It cannot authorize
            # descent or release; a contradictory condition never permits motion.
            conditional_keys=[r['condition'] for r in self.task_definition['subgoals'] if r['object'] in conditional]
            missing_conditions=[key for key in unknown if key not in conditional_keys]
            if failed or missing_conditions:
                keys=failed+missing_conditions
                self._stop('active_task_condition_requires_revalidation',
                    [self.conditions.get(key)['id'] for key in keys if self.conditions.get(key)['id']],
                    metrics={'failed_conditions':keys,'recoverable':True,
                             'next_action':'restore_task_prerequisite_before_remaining_motion'})
        policy='conditional_history_clear_transit' if conditional else 'current_localization'
        self.graph._event('active_dependency_revalidation',plan_id=plan_id,
            source_ids=contract['sources'],expected_external_epochs=contract['external_epochs'],
            started=contract['started'],stage=contract.get('stage'),
            status='UNCERTAIN' if missing else 'PASS',evidence_policy=policy,
            missing_objects=missing,conditional_objects=conditional,
            independent_verification=False)
        if missing:
            self.record('L2','active_dependencies','UNCERTAIN','current_support_evidence_required',
                {'plan_id':plan_id,'stage':contract.get('stage'),'missing_objects':missing},action='reobserve')
            self._stop('current_support_evidence_required',[plan_id,self.last_check],
                metrics={'recoverable':True,'missing_surface_object':missing[0],
                         'next_action':'acquire_support_evidence_then_new_place_plan'})
        if self.memory and motion is not None and contract.get('stage')=='final':
            # A predicted surface is not enough by itself: the intended final
            # pose must satisfy the same release geometry over the full error
            # interval before descending. Both policies use this common gate.
            data=self.graph.nodes[plan_id]['data']
            if data['skill']=='place':
                obj=self.graph.nodes[data['inputs']['support_surface']]['data']['object']
                support=self._current_state_surface(obj);held=self._held_surface()
                if any((x or {}).get('evidence_kind')=='predicted' for x in (support,held)):
                    if not held or not support:self._stop('prediction_missing_final_descent_evidence')
                    from .evidence_planning import placement_obligations
                    if support.get('evidence_kind')!='predicted':support=self._calibrated_direct_surface(obj)
                    obligation=placement_obligations(self.memory,held_object=self.held_object,support=support,
                        geometry=self.geometry[self.held_object],tcp=self.last_public['robot_cartesian_pos'],
                        target=motion[0],quaternion=motion[1],stamp=time.monotonic(),
                        calibration=self._memory_calibration(),grasp_epoch=self.grasp_epoch)
                    check=obligation.get('release_forecast',obligation)
                    self.record('L2','predicted_descent',check['status'],check['reason'],check)
                    if check['status']!='PASS':self._stop('prediction_interval_blocks_final_descent',metrics={'recoverable':True})

    def _stop(self,reason,roots=(),incomplete=False,metrics=None):
        # Preserve the actual failed check as a causal root before adding L3.
        return super()._stop(reason,list(roots) or list(self.roots),incomplete,metrics)

    def _new(self,key,data,parents=()):
        old=self.graph.latest.get(key)
        if old and self.graph.nodes[old]['valid']:
            roots=[old]
            if not self.graph_options['selective_invalidation']:
                # A routine read replaces this source and G0's unexecuted plans,
                # not sibling measurements or completed-action provenance. A
                # consumed plan is history: rooting it again would propagate
                # through immutable facts into freshly measured held surfaces.
                roots += [n['id'] for n in self.graph.nodes.values()
                          if n['kind']=='plan' and n['valid']
                          and (n['data'] or {}).get('execution')!='consumed']
            self.graph.invalidate(roots,'public_source_replaced:'+key)
        node=self.graph.version(key,'estimate',data,parents=list(parents)+([self.last_observation] if self.last_observation else []),invalidate=False)
        self.change_roots=[node]
        if self.conditions:self.conditions.api_source(node)
        return {'id':node,**copy.deepcopy(data)}

    def _surface(self,name):
        return self.last_evidence.get(self._object_binding(name)['sensor_key'])

    def _observe_object(self,name,fn):
        self._object_binding(name)
        value=FeedbackMonitor.invoke(self,'get_object_pose',[self._object_binding(name)['perception_name']],{'return_bbox_extent':True},fn)
        self.capture();surface=self._surface(name)
        if not self.closed or name!=self.held_object:
            # A nonheld object's current public geometry can be acquired during
            # recovery. Never recalibrate the held object's size from occlusion.
            geometry=object_geometry(name,surface,self.table_z,self.last_evidence,self.geometry.get(name))
            if geometry:
                # A geometric prior measured before approach; never captured only after close occlusion.
                geometry['orientation_verified']=False
                self.geometry[name]=geometry
                self.geometry_ids[name]=self.graph.add('verification_fact',{'type':'public_geometry_calibration',**geometry},parents=[self.last_observation])
        p,q,e=value
        return self._new('object:'+name,dict(type='object',object=name,position=np.asarray(p).tolist(),
            quaternion_wxyz=np.asarray(q).tolist(),quaternion_order='wxyz',extent=np.asarray(e).tolist(),
            frame='robot_base',object_epoch=self.object_epochs[name]),[self.last_call])

    def _robot(self):
        state,_=self.capture();self._reconcile_context();tcp=np.asarray(state['robot_cartesian_pos'])
        return self._new('robot_read',dict(type='robot',frame='robot_base',quaternion_order='wxyz',
            robot_epoch=self.robot_epoch,position=tcp[:3].tolist(),quaternion_wxyz=tcp[3:7].tolist(),
            tcp=tcp.tolist(),joints=np.asarray(state['robot_joint_pos']).tolist(),
            gripper_joints=None if 'robot_gripper_joint_pos' not in state else np.asarray(state['robot_gripper_joint_pos']).tolist()))

    def _surfaces(self,name,fn=None):
        self._object_binding(name)
        self.capture();s=self._held_surface() if name==self.held_object and self.closed else self._current_state_surface(name) if self.memory else self._support_surface(name)
        if s is None and name==self.held_object and self.closed and fn is not None:
            s=self._recover_held_visibility(fn)
        if s is None:
            support_occluded=self.closed and self.phase=='holding_verified' and name!=self.held_object
            self._stop('current_object_surface_unavailable',metrics={'object':name,
                'missing_surface_object':name,'perception_failure_behavior':'actor_interruption_not_None',
                'recoverable':self.terminal_reason is None and self._view_budget()['change_view_available'],
                'next_action':'replan_observation_pose_before_retrying_support' if support_occluded else 'inspect_public_clearance_or_stop',
                **self._view_budget()})
        return self._new('surface:'+name,{**s,'type':'surface','object':name,'object_epoch':self.object_epochs[name]},
                         ([self.geometry_ids[name]] if name in self.geometry_ids else [])+([self.attachment_id] if self.attachment_id and name==self.held_object else [])+([s['prior_source_id']] if s.get('prior_source_id') else []))

    def _view_context(self,tcp,*,translated=False):
        """Explicit policy state; current public pose is never a geometry permit."""
        tcp=np.asarray(tcp,float)
        current=self.visibility_context
        if current is not None and current['grasp_epoch']!=self.grasp_epoch:current=None
        if current is not None and not translated:return current
        if current is not None and np.linalg.norm(tcp[:3]-np.asarray(current['anchor_tcp_xyz']))<self.visibility_translation_m:return current
        matches=[c for c in self.visibility_contexts if c['grasp_epoch']==self.grasp_epoch
            and np.linalg.norm(tcp[:3]-np.asarray(c['anchor_tcp_xyz']))<self.visibility_translation_m]
        if matches:
            current=min(matches,key=lambda c:np.linalg.norm(tcp[:3]-np.asarray(c['anchor_tcp_xyz'])))
        else:
            current={'context_id':'view-'+str(len(self.visibility_contexts)+1),
                'grasp_epoch':self.grasp_epoch,'anchor_tcp_xyz':tcp[:3].tolist(),
                'reference_quaternion_wxyz':tcp[3:7].tolist(),'tried_yaw_degrees':[]}
            parents=[x for x in (self.last_observation,self.last_call) if x]
            current['source_id']=self.graph.add('verification_fact',{
                'type':'visibility_context_policy','historical_only':True,'permission':False,
                'min_translation_m':self.visibility_translation_m,**copy.deepcopy(current)},parents=parents)
            self.visibility_contexts.append(current)
            self.graph._event('visibility_context_created',context=copy.deepcopy(current))
        if self.visibility_context is not current:
            self.graph._event('visibility_context_selected',context_id=current['context_id'],grasp_epoch=self.grasp_epoch)
        self.visibility_context=current
        self.visibility_reference_q=np.asarray(current['reference_quaternion_wxyz'],float)
        return current

    def _view_after_motion(self,state,kind):
        # A capture or a yaw/hold probe must not replenish candidates. Only a
        # returned ordinary motion may establish another public TCP location.
        if self.closed and self.phase=='holding_verified' and self.attachment is not None and kind not in ('visibility_probe','hold_probe','hold_probe_return'):
            self._view_context(state['robot_cartesian_pos'],translated=True)

    def _view_budget(self):
        context=self.visibility_context
        if context is not None and context['grasp_epoch']!=self.grasp_epoch:context=None
        available=[x for x in (20,-20) if context is None or x not in context['tried_yaw_degrees']]
        remaining=max(0,self.max_episode_views-self.visibility_probes)
        local=max(0,self.max_episode_local-self.local_attempts)
        return {'remaining_visibility_probes':remaining,'remaining_local_attempts':local,
            'change_view_available':bool(remaining and local and available and self.phase=='holding_verified' and self.attachment is not None),
            'visibility_context':None if context is None else {**copy.deepcopy(context),'available_yaw_degrees':available},
            'visibility_policy':{'max_episode_views':self.max_episode_views,'min_translation_m':self.visibility_translation_m,
                'yaw_candidates_degrees':[20,-20],'context_requires_returned_nonprobe_translation':True}}

    def _recover_held_visibility(self,fn,*,single_step=False,verification_probe=False,require_anchor=False):
        """Bounded active perception; no simulated object state or blind release.

        A wrist-yaw changes the camera view without repeatedly raising the object.
        Clearance is conditional on public rigid attachment and visible known
        neighbours. This is not an obstacle-free arm-path certificate.
        """
        verifying=(verification_probe is True and self.phase=='holding_unverified'
            and self.closed and self.lift_attempts>=1)
        if (self.phase!='holding_verified' and not verifying) or self.attachment is None:return None
        if self.visibility_probes>=self.max_episode_views:
            self.terminal_reason='visibility_probe_budget_exhausted';return None
        for _ in range(1 if single_step else 2):
            state,_=self.capture();tcp=np.asarray(state['robot_cartesian_pos'],float)
            context=self._view_context(tcp)
            candidates=[x for x in (20,-20) if x not in context['tried_yaw_degrees']]
            if not candidates or self.visibility_probes>=self.max_episode_views:break
            predicted=predict_attachment(self.attachment,tcp)
            bottom=float(predicted['bottom_z']);center=np.asarray(predicted['center'])
            clear=bool(predicted.get('orientation_verified') and self.table_z is not None
                       and bottom-float(self.table_z)>=.05 and tcp[7]>.05)
            if verifying:
                clear=clear and self.close_tcp_z is not None and tcp[2]-self.close_tcp_z>=.05
            neighbor_evidence=[]
            for obj in [x for x in self.object_epochs if x!=self.held_object]:
                points=self.colored_points.get(obj,np.empty((0,3)))
                surface=self._surface(obj)
                known=surface is not None and len(points)>=30
                near=known and np.linalg.norm(np.asarray(surface['center'])[:2]-center[:2])<.15
                safe=known and (not near or bottom-float(np.quantile(points[:,2],.99))>=.03)
                neighbor_evidence.append({'object':obj,'visible':known,'near':bool(near),'safe':bool(safe)})
                clear=clear and safe
            self.record('L2','visibility_probe_clearance','PASS' if clear else 'UNCERTAIN',
                'public_clearance_for_bounded_yaw' if clear else 'active_view_clearance_unverified',
                {'bottom_z':bottom,'table_z':self.table_z,'neighbors':neighbor_evidence,
                 'purpose':'acquire_evidence_for_hold_verification' if verifying else 'refresh_verified_hold_view',
                 'view_does_not_verify_grasp':True,'conditional_rigid_attachment':True,'full_arm_collision_proof':False,**self._view_budget()},
                graph_parents=[context['source_id']])
            if not clear:return None
            if self.local_attempts>=self.max_episode_local:
                self.terminal_reason='local_recovery_budget_exhausted';return None
            yaw=candidates[0]
            rotation=Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_quat(np.roll(self.visibility_reference_q,-1))
            target_q=np.roll(rotation.as_quat(),1).tolist()
            self._phase_guard('visibility_probe',True,'public_clearance_allows_active_view')
            context['tried_yaw_degrees'].append(yaw)
            self.visibility_probes+=1;self.local_attempts+=1;self.recoveries+=1
            self.graph._event('visibility_candidate_reserved',context_id=context['context_id'],yaw_degrees=yaw,
                attempt=self.visibility_probes,remaining_episode=max(0,self.max_episode_views-self.visibility_probes))
            self._move(tcp[:3].tolist(),target_q,0.,fn,'visibility_probe')
            self.capture();surface=self._held_surface()
            if require_anchor:
                from .table_clearance import _current_cube_shape
                if not _current_cube_shape(surface) or surface.get('evidence_kind')=='predicted':surface=None
            self.recovery_outcomes.append({'kind':'active_visibility','attempt':self.visibility_probes,
                'yaw_from_reference_degrees':yaw,'context_id':context['context_id'],
                'success':surface is not None,'public_only':True})
            if surface is not None or single_step:return surface
        self.terminal_reason='visibility_context_candidates_exhausted' if single_step else 'held_surface_evidence_unrecoverable'
        return None

    def _prepare_evidence(self,fn):
        from .table_clearance import _current_cube_shape
        allowed=bool(self.memory and self.phase=='holding_verified' and self.closed and self.lifted)
        self._phase_guard('prepare_evidence',allowed,'verified_hold_required_for_geometry_anchor')
        self.capture();surface=self._held_surface()
        direct=lambda s: bool(_current_cube_shape(s) and s.get('evidence_kind')!='predicted')
        before=direct(surface)
        if not before:surface=self._recover_held_visibility(fn,single_step=True,require_anchor=True)
        good=direct(surface)
        self.record('L0','future_anchor','PASS' if good else 'UNCERTAIN',
            'identifiable_current_anchor' if good else 'no_new_identifiable_geometry',
            {'missing_quantity':'held_geometry_anchor','progress':good and not before,
             'new_source_id_alone_is_progress':False,'future_request':self.evidence_request,
             'physical_view_budget':self._view_budget()},action='continue' if good else 'reobserve')
        if not good:self._stop('future_anchor_unavailable',metrics={'recoverable':self._view_budget()['change_view_available'],
            'next_action':'prepare_evidence_then_fresh_plan','missing_evidence':['held_geometry_anchor']})
        self.evidence_request=None
        return dict(status='PASS',evidence_id=self.last_check,permission=False)

    def _prepare_descent(self,p,q):
        """Resolve future evidence while the object is still above the target."""
        if not self.evidence_planning or not self.memory or not self.active_plan:return
        from .evidence_planning import placement_obligations
        from .table_clearance import _current_cube_shape
        d=self.graph.nodes[self.active_plan]['data'];obj=d['object']
        self.capture();held=self._held_surface()
        intent=self.conditions.intent(d['skill'],obj) if self.conditions else None
        if d['skill']=='place':
            support_obj=self.graph.nodes[d['inputs']['support_surface']]['data']['object']
            support=self._current_state_surface(support_obj)
            if support and support.get('evidence_kind')!='predicted':support=self._calibrated_direct_surface(support_obj)
            row=placement_obligations(self.memory,held_object=obj,support=support,
                geometry=self.geometry[obj],tcp=self.last_public['robot_cartesian_pos'],target=p,quaternion=q,
                stamp=time.monotonic(),calibration=self._memory_calibration(),grasp_epoch=self.grasp_epoch,intent_id=intent)
        else:
            # Table placement uses its existing CURRENT instance/robot/ray
            # clearance contract. A predicted box cannot replace those masks.
            validation=self._current_table_check(d['inputs']['table_region'])
            row=dict(status='PASS',reason='current_table_contract_before_descent',permission=False,
                future_intent_id=intent,requirements=['held_instance_shape','fresh_table_clearance','fresh_final_capture'],
                missing=[],current_validation=validation,next_work='continue_to_boundary_checks')
        parents=[self.active_plan,self.last_observation]+([intent] if intent else [])
        parents+=row.get('retained_sources',[])
        node=self.graph.add('evidence_plan',row,parents=parents)
        self.record('L0','future_evidence',row['status'],row['reason'],row,graph_parents=[node])
        if row['status']!='PASS':
            self.evidence_request=dict(row,source_id=node)
            self._stop('future_evidence_not_ready',[node,self.last_check],metrics={'recoverable':True,
                'missing_evidence':row['missing'],'next_action':'prepare_evidence_then_fresh_plan'
                    if row['next_work']=='acquire_anchor' else 'fresh_alignment_plan_from_current_surfaces'})

    def _reserve_semantic_refinement(self):
        # Perception recovery consumes the SAME episode-local budget as motion
        # recovery. Actors/captures cannot replenish it; reservation precedes HTTP.
        if self.local_attempts>=self.max_episode_local:return False
        self.local_attempts+=1;self.recoveries+=1;self.perception_refinements+=1
        self.graph._event('semantic_refinement_reserved',attempt=self.perception_refinements,
            observation_id=self.last_observation,remaining_local_attempts=max(0,self.max_episode_local-self.local_attempts))
        return True

    def _table_evaluation(self,region,parents=(),region_id=None):
        """Fresh shared evidence; historical bounds alone never grant permission."""
        self.capture();held=self._held_surface()
        if self.table_evidence_fn is None or held is None:
            result={'validation':{'status':'UNCERTAIN','reason':'current_held_or_table_provider_unavailable','clearance_verified':False},'evidence':{}}
        else:
            geometry={name:{**value,'source_id':self.geometry_ids.get(name)} for name,value in self.geometry.items()}
            result=self.table_evidence_fn(self.current_public_state,region,self.held_object,held,
                observation_id=self.last_observation,held_identity_verified=self.phase=='holding_verified',nonheld_geometry=geometry)
        validation=result['validation'];evidence=result['evidence']
        parents=list(parents)+([self.attachment_id] if self.attachment_id else [])
        if held and held.get('memory_id'):parents.append(held['memory_id'])
        if held and held.get('attachment_check_id'):parents.append(held['attachment_check_id'])
        parents.extend(evidence.get('geometry_source_ids',[]));parents=list(dict.fromkeys(parents))
        for source in parents:self._valid(source)
        refinement=evidence.get('semantic_refinement',{})
        if refinement.get('attempted'):
            self.record('L0','semantic_refinement',validation['status'],'current_crop_semantic_reobservation',
                {**refinement,'final_validation':validation,'physical_motion':False},graph_parents=parents,
                action='continue' if validation['status']=='PASS' else 'reobserve')
            self.recovery_outcomes.append({'kind':'semantic_refinement','attempt':self.perception_refinements,
                'success':validation['status']=='PASS','public_only':True,'physical_motion':False})
        self.record('L2','table_clearance',validation['status'],validation['reason'],
            {'validation':validation,'evidence':evidence,'region_id':region_id},
            graph_parents=parents,action='continue' if validation['status']=='PASS' else 'reobserve')
        return validation,self.last_check

    def _remember_unverified_table_target(self,region_id,gate):
        """Search memory only: unclear geometry never becomes free-space proof.

        A completed move stays completed. This check issues no physical action;
        neither the whole plan nor earlier calls are labelled not-executed.
        """
        validation=self.graph.nodes[gate]['data']['metrics'].get('validation',{})
        if validation.get('status')!='FAIL' or validation.get('reason')!='current_nonheld_obstacle_in_table_region':return None
        if self._table_needs_identity_view(gate):return None
        if self.phase!='holding_verified' or not self.closed or self.terminal_reason is not None:return None
        region=copy.deepcopy(self.graph.nodes[region_id]['data'])
        if region.get('object')!=self.held_object:return None
        self.last_place=None
        old=next((r for r in self.rejected_table_targets if r['region_id']==region_id
            and r['object']==self.held_object and r['grasp_epoch']==self.grasp_epoch),None)
        if old:return old['source_id']
        fact=self.graph.add('verification_fact',{'type':'unverified_table_target',
            'historical_only':True,'region_id':region_id,'source_check_id':gate,
            'object':self.held_object,'grasp_epoch':self.grasp_epoch,
            'physical_obstacle_confirmed':False,'physical_action_issued_by_this_check':False,
            'scope':'current_grasp_search_exclusion_not_occupancy_or_global_unreachability',
            'inflight':copy.deepcopy(self.inflight)},parents=[region_id,gate])
        self.rejected_table_targets.append({'region_id':region_id,'source_id':fact,
            'object':self.held_object,'grasp_epoch':self.grasp_epoch,'region':region})
        return fact

    def _table_needs_identity_view(self,gate):
        # When all named nonheld objects are CURRENTLY localized outside this
        # region, leftover depth has unknown identity. One existing legal view
        # is more informative than declaring another region necessary. This
        # does not clear any pixel, change FAIL, or authorize a physical action.
        evidence=self.graph.nodes[gate]['data']['metrics'].get('evidence',{})
        rows=evidence.get('nonheld_current_evidence',[])
        expected=set(self.object_epochs)-{self.held_object}
        established={r.get('object') for r in rows if r.get('source_id')
            and r.get('outside_region_with_margin') is True}
        return bool(expected and expected<=established and self._view_budget()['change_view_available'])

    def _table_recovery_hint(self,gate):
        """Choose evidence acquisition before view motion when a source is missing."""
        metrics=self.graph.nodes[gate]['data']['metrics']
        validation=metrics.get('validation',{})
        evidence=metrics.get('evidence',{})
        missing=sorted({row['object'] for row in evidence.get('nonheld_current_evidence',[])
            if row.get('object') in self.object_epochs and row['object']!=self.held_object
            and not row.get('source_id')})
        hint={'recoverable':self.terminal_reason is None and self._view_budget()['change_view_available'],
              'next_action':'reobserve_table_same_region_then_new_plan_or_stop',**self._view_budget()}
        if (missing and validation.get('status')=='UNCERTAIN'
            and validation.get('reason')=='current_table_region_occluded_or_insufficiently_visible'):
            hint.update(recoverable=self.terminal_reason is None,missing_geometry_objects=missing,
                next_action='observe_missing_nonheld_objects_then_reobserve_table_without_motion',
                reobserve_change_view=False)
        if (validation.get('status')=='FAIL'
            and validation.get('reason')=='current_nonheld_obstacle_in_table_region'
            and not self._table_needs_identity_view(gate)):
            hint.update(recoverable=self.terminal_reason is None,
                next_action='observe_alternative_table_region_then_refresh_surface_robot_and_plan',
                rejected_region_id=metrics.get('region_id'))
        return hint

    def _current_table_check(self,region_id):
        """Current observation authorizes this boundary, never a historic flag."""
        region=self._valid(region_id)['data']
        validation,gate=self._table_evaluation(region,[region_id],region_id)
        if validation['status']!='PASS':
            self.graph.invalidate([region_id],'table_clearance_changed_or_unverified',conservative=not self.graph_options['selective_invalidation'])
            history=self._remember_unverified_table_target(region_id,gate)
            self._stop('table_clearance_'+validation['reason'],[region_id,gate]+([history] if history else []),metrics={'region_id':region_id,**self._table_recovery_hint(gate)})
        self._valid(gate)
        if self.active_authorization:
            self.active_authorization=self.graph.add('verification_fact',{
                'historical_only':True,'type':'boundary_table_authorization',
                'source_check_id':gate,'region_id':region_id,'block':self.block},
                parents=[self.active_authorization,(gate,'precondition')])
        return validation

    def _reobserve_table(self,region_id,change_view=True,*,fn):
        """Use an old region only as location intent and issue a fresh permit."""
        if type(change_view) is not bool or not isinstance(region_id,str):raise ValueError('Invalid table reobservation arguments')
        old=self.graph.nodes.get(region_id)
        if old is None or old['kind']!='estimate' or old['data'].get('type')!='table_region':
            self._stop('known_table_region_intent_required',metrics={'recoverable':False})
        region=copy.deepcopy(old['data'])
        allowed=self.phase=='holding_verified' and self.closed and self.lifted and region.get('object')==self.held_object and self.attachment is not None
        self._phase_guard('reobserve_table',allowed,'verified_matching_held_object_required' if not allowed else 'current_held_phase_allows_table_reobservation')
        # An invalid historical source can be recorded as intent, not inherited
        # as a current permit. A new check must establish all current evidence.
        self.graph.invalidate([region_id],'table_reobservation_requested',conservative=not self.graph_options['selective_invalidation'])
        self.last_place=None
        intent=self.graph.add('verification_fact',{'type':'table_region_reobservation_intent',
            'historical_only':True,'permission':False,'prior_region_id':region_id,
            'object':region['object'],'center':region['center'],'lo':region['lo'],'hi':region['hi'],
            'grasp_epoch':self.grasp_epoch,'change_view':change_view},parents=[region_id]+([self.fragment] if self.fragment else []))
        budget=self._view_budget
        if change_view:
            if self.visibility_probes>=self.max_episode_views:
                self.terminal_reason='visibility_probe_budget_exhausted'
                self._stop(self.terminal_reason,[intent],metrics={'recoverable':False,**budget()})
            before=self.visibility_probes
            held=self._recover_held_visibility(fn,single_step=True)
            if self.visibility_probes==before or held is None:
                self._stop('table_active_view_unavailable',[intent]+([self.last_check] if self.last_check else []),
                    metrics={'recoverable':self.terminal_reason is None and budget()['change_view_available'],**budget()})
        validation,gate=self._table_evaluation(region,[intent],region_id)
        if validation['status']!='PASS':
            history=self._remember_unverified_table_target(region_id,gate)
            self._stop('table_clearance_'+validation['reason'],[intent,gate]+([history] if history else []),
                metrics={'region_id':region_id,**self._table_recovery_hint(gate)})
        self._valid(gate)
        refreshed={**region,**{k:v for k,v in validation.items() if k in ('clearance_verified','visible_plane_points','footprint_coverage_fraction','center_coverage_verified','conditional_static_scene')},
            'source':'current_RGBD_revalidated_table_intent','prior_region_id':region_id,
            'reobservation_intent_id':intent,'historical_clearance_check_id':gate,
            'clearance_permission':'fresh_boundary_check_required',
            'current_observation_id':self.last_observation}
        if self.table_z is not None:
            for key in ('center','lo','hi'):refreshed[key]=[*region[key][:2],float(self.table_z)]
            refreshed['top_z']=float(self.table_z)
        # Region geometry/intent remains data, not a lasting permission. Keep
        # the successful check ID for diagnosis, but do not make the geometry
        # expire when G0 retires that one-boundary check after its own motion.
        # The current RGB-D observation (and its calibration) remains a real
        # parent via _new; every physical segment rechecks current clearance.
        result=self._new('table_region',refreshed,[intent])
        self._valid(result['id'])
        self.recovery_outcomes.append({'kind':'table_reobservation','prior_region_id':region_id,
            'new_region_id':result['id'],'change_view':change_view,'success':True,**budget()})
        return result

    def _current_camera_origin(self):
        cameras=[value for value in (self.current_public_state or {}).values() if isinstance(value,dict) and 'images' in value and 'pose_mat' in value]
        return np.asarray(cameras[0]['pose_mat'],float)[:3,3] if cameras else None

    def _support_surface(self,name,allow_retained=True):
        visible=self._surface(name);geom=self.geometry.get(name)
        if not geom:
            geom=object_geometry(name,visible,self.table_z,self.last_evidence)
            if geom:
                self.geometry[name]=geom
                self.geometry_ids[name]=self.graph.add('verification_fact',{'type':'public_support_size',**geom},parents=[self.last_observation])
        if reliable_top(visible,geom):
            return measured_surface(visible,geom)
        points=self.colored_points.get(name,np.empty((0,3)))
        current=current_support(points,self.table_z,geom,camera_origin=self._current_camera_origin())
        if current:return current
        old=self.support_history.get(name)
        if allow_retained and old and old['epoch']==self.object_epochs[name] and retained_consistent(points,old['surface'],geom):
            return {**copy.deepcopy(old['surface']),'frame':'robot_base','source':'retained_public_support_pose',
                    'prior_source_id':old['source_id'],'object_epoch':old['epoch'],
                    'full_current_localization':False,'limitation':'historical pose with no registered intervening contact and current visible-point consistency; hidden motion remains possible'}
        return None

    def _held_surface(self):
        visible=self._surface(self.held_object)
        if self.attachment is None:return measured_surface(visible,self.geometry.get(self.held_object))
        predicted=predict_attachment(self.attachment,self.last_public['robot_cartesian_pos'])
        check=check_attachment(predicted,visible,points=self.colored_points.get(self.held_object),geometry=self.geometry.get(self.held_object),camera_origin=self._current_camera_origin())
        self.record('L0','attachment_consistency',check['status'],check['reason'],check,
                    graph_parents=[self.attachment_id])
        if check['status']!='PASS':
            if check['status']=='FAIL' and self.memory:
                self.memory.contradict(self.held_object,visible,source=self.last_observation)
            return self._predicted_surface(self.held_object) if check['status']=='UNCERTAIN' else None
        return {**predicted,**check.get('current_surface',{}),'frame':'robot_base','attachment_check_id':self.last_check,
                'conditional_attachment':True,'visible_surface':visible}

    def _memory_calibration(self):
        # Stable semantic camera projection, not ephemeral graph node IDs.
        return hashlib.sha256(json.dumps([
            self.graph.nodes[i]['data']['calibration'] for i in self.observation_calibration_sources
        ],sort_keys=True).encode()).hexdigest() if self.observation_calibration_sources else None

    def _remember_surface(self,obj,surface,attached):
        if not self.memory or not surface:return
        self.memory.observe(obj,surface,self.geometry[obj],source=self.last_observation,
            calibration=self._memory_calibration(),stamp=time.monotonic(),evidence_ref=self.evidence_reference,
            tcp=self.last_public['robot_cartesian_pos'],attached=attached,grasp_epoch=self.grasp_epoch)

    def _refresh_memory(self):
        """Memory never turns a consistency heuristic into independent evidence."""
        for obj,geom in self.geometry.items():
            attached=self.closed and obj==self.held_object
            visible=self._surface(obj);prior=self.memory.records.get(obj)
            if attached:
                if not self.attachment or not self.lifted:continue
                predicted=predict_attachment(self.attachment,self.last_public['robot_cartesian_pos'])
                check=check_attachment(predicted,visible,points=self.colored_points.get(obj),
                    geometry=geom,camera_origin=self._current_camera_origin())
                if check['status']=='FAIL':self.memory.contradict(obj,visible,source=self.last_observation)
                elif check['status']=='PASS' and check.get('current_surface'):
                    self._remember_surface(obj,{**predicted,**check.get('current_surface',{}),'frame':'robot_base'},True)
                elif prior:
                    # Account for ALL motion, including out-and-back paths;
                    # missing pixels cannot reset age, distance or rotation.
                    if self.memory.partial_conflict(obj,self.colored_points.get(obj,[]),self.last_public['robot_cartesian_pos']):
                        self.memory.contradict(obj,visible,source=self.last_observation)
                    else:self._predicted_surface(obj)
            else:
                current=current_support(self.colored_points.get(obj,np.empty((0,3))),self.table_z,geom,
                    camera_origin=self._current_camera_origin())
                if current:
                    if prior and prior['status'] not in ('UNKNOWN','CONTRADICTED'):
                        delta=np.linalg.norm(np.asarray(current['center'])[:2]-np.asarray(prior['surface']['center'])[:2])
                        dz=abs(current['hi'][2]-prior['surface']['hi'][2])
                        if delta>self.memory.profile['xy_m']*2 or dz>self.memory.profile['z_m']*2:
                            self.memory.contradict(obj,current,source=self.last_observation)
                            # Fresh independent localization starts a NEW version;
                            # its old dependent plans are already invalidated.
                    self._remember_surface(obj,current,False)
                elif prior:
                    # Assess loss of the noncontact assumption even when no
                    # caller currently asks for this support. A later ascent
                    # must not revive evidence from before a possible touch.
                    self._predicted_surface(obj)

    def _predicted_surface(self,obj):
        if not self.memory:return None
        from .state_memory import depth_occlusion
        r=self.memory.records.get(obj)
        if not r:return None
        surface=copy.deepcopy(r['surface']);tcp=self.last_public['robot_cartesian_pos']
        if r['attached']:
            from .grip_offset import rotation
            delta=np.asarray(tcp[:3])+rotation(tcp[3:7])@np.asarray(r['offset'])-np.asarray(surface['center'])
            for key in ('center','lo','hi'):surface[key]=(np.asarray(surface[key])+delta).tolist()
        occlusion=depth_occlusion(self.current_public_state or {},surface,r['geometry']['height_m'])
        # Static prediction needs positive clearance, not merely absence of a
        # contact sensor. Release/near-contact invalidates it explicitly.
        clear=self._static_memory_clear(obj,surface)
        estimate=self.memory.estimate(obj,tcp=tcp,stamp=time.monotonic(),calibration=self._memory_calibration(),
            grasp_epoch=self.grasp_epoch,hold_verified=self.phase=='holding_verified' and self.lifted,
            occlusion=occlusion,static_clear=clear)
        if estimate['status']!='PREDICTED':
            self.graph._event('memory_rejected',object=obj,**estimate)
            return None
        node=self.graph.add('state_prediction',estimate,parents=[estimate['source_id'],self.last_observation])
        return dict(estimate['surface'],frame='robot_base',evidence_kind='predicted',
            memory_id=node,error_bounds=estimate['bounds'],current_visual_detection=False,
            evidence_ref=estimate['evidence_ref'],prior_source_id=estimate['source_id'])

    def _static_memory_clear(self,obj,surface):
        if not self.closed or not self.held_object or not self.attachment:return False
        if obj==self.held_object:return False
        from .grip_offset import rotation
        record=self.memory.records.get(self.held_object)
        if not record or not record['attached'] or record['status'] in ('UNKNOWN','CONTRADICTED'):return False
        tcp=np.asarray(self.last_public['robot_cartesian_pos'])
        center=tcp[:3]+rotation(tcp[3:7])@np.asarray(record['offset'])
        bottom=center[2]-record['geometry']['height_m']/2
        p=self.memory.profile
        # For support below a rigid carried cube, a certified positive gap is
        # the only first-version noncontact witness. Other configurations stay
        # UNKNOWN; no inference from missing contact readings.
        gap=bottom-surface['hi'][2]
        clear=gap>2*p['z_m']+p['height_m'] and self.phase=='holding_verified'
        if not clear:self.memory.invalidate(obj,'possible_contact')
        return clear

    def _current_state_surface(self,obj):
        # Target compilation and its future verifier must consume the SAME
        # current estimator. Replanning from a loose top centroid while checking
        # against an identifiable face simply repeats a deterministic bias.
        calibrated=self._calibrated_direct_surface(obj)
        if calibrated is not None:return calibrated
        current=self._support_surface(obj,allow_retained=False)
        return current if current is not None else self._predicted_surface(obj)

    def _calibrated_direct_surface(self,obj):
        """Only this capture's identifiable face fit has the face envelope.

        The legacy full-top/attachment screen remains a direct-observation path,
        but may not borrow a narrower calibration for a different estimator.
        SAM3 shape refinement has not been calibrated as a memory anchor either.
        """
        if not self.memory:return None
        record=self.memory.records.get(obj)
        if not record or record['current_status']!='DIRECT':return None
        node=self.graph.nodes[record['node']]
        if not node['valid'] or node['data'].get('source_id')!=self.last_observation:return None
        if record['calibration']!=self._memory_calibration():return None
        return dict(record['surface'],evidence_kind='direct_calibrated_face',
            current_visual_detection=True,calibrated_observation_id=self.last_observation,
            calibration_model_id=self.memory.model,memory_anchor_ref=record['node'],
            evidence_ref=record['evidence_ref'],
            error_bounds={k:self.memory.profile[k] for k in ('xy_m','z_m','height_m')})

    def _grasp(self,name,fn):
        if self.closed:self._stop('already_holding_do_not_resample_grasp')
        if name not in self.geometry:self._observe_object(name,fn)
        raw=FeedbackMonitor.invoke(self,'sample_grasp_pose',[self._object_binding(name)['perception_name']],{},fn)
        self.capture();s=self._surface(name);g=self.geometry.get(name)
        if not s or not g:self._stop('pregrasp_geometry_unavailable',metrics={'object':name,'recoverable':'reobserve_before_approach'})
        q=np.asarray(raw[1],float)
        conversion=cgn_upright_panda(q);upright=conversion['quaternion_wxyz']
        p=[s['center'][0],s['center'][1],s['hi'][2]-g['height_m']/2]
        source=self.graph.add('verification_fact',dict(type='raw_grasp_candidate',object=name,
            position=np.asarray(raw[0]).tolist(),quaternion_wxyz=q.tolist()),parents=[self.last_call,self.last_observation])
        raw_rotation=Rotation.from_quat(np.roll(q,-1)).as_matrix()
        output_rotation=Rotation.from_quat(np.roll(upright,-1)).as_matrix()
        converted=self.graph.add('verification_fact',dict(type='gripper_frame_conversion',object=name,
            **conversion,raw_rotation_base=raw_rotation.tolist(),output_rotation_base=output_rotation.tolist(),
            raw_to_output_right_rotation=(raw_rotation.T@output_rotation).tolist(),
            upright_projection=True,collision_verified=False),parents=[source,self.last_observation])
        return self._new('grasp:'+name,dict(type='grasp',object=name,frame='robot_base',quaternion_order='wxyz',
            position=p,quaternion_wxyz=upright,object_epoch=self.object_epochs[name],reference_surface=copy.deepcopy(s),
            fusion='visible_cube_center_CGN_x_to_Panda_y_upright',raw_candidate_id=source,
            conversion_id=converted),[source,converted,self.geometry_ids[name]])

    def _plan_skill(self,skill,inputs,params=None):
        if not isinstance(inputs,dict) or not all(isinstance(v,str) for v in inputs.values()):
            self._stop('parent_source_ids_required',incomplete=True,metrics={'usage':"Use observe_result['id'] strings in inputs; never pass the complete dictionary"})
        sources={role:self._valid(value) for role,value in inputs.items()}
        types={'robot':'robot','grasp':'grasp','held_surface':'surface','support_surface':'surface','table_region':'table_region'}
        if any(role not in types or node['data'].get('type')!=types[role] for role,node in sources.items()):
            self._stop('source_role_type_mismatch',list(inputs.values()))
        if skill in ('place','table_place') and 'held_surface' in sources and 'bottom_z' not in sources['held_surface']['data']:
            self._stop('verified_upright_height_required',list(inputs.values()))
        target=compile_target(skill,sources,params)
        if skill=='approach':obj=sources['grasp']['data']['object']
        else:obj=self.held_object
        if skill in ('place','table_place') and sources['held_surface']['data'].get('object')!=self.held_object:
            self._stop('held_surface_identity_mismatch',list(inputs.values()))
        if skill=='place' and sources['support_surface']['data'].get('object')==self.held_object:
            self._stop('support_must_be_a_different_object',list(inputs.values()))
        if skill=='table_place' and sources['table_region']['data'].get('object')!=self.held_object:
            self._stop('table_region_object_identity_mismatch',list(inputs.values()))
        parents=[(i,'data') for i in inputs.values()]+[(self.capability,'precondition')]
        if self.fragment:parents.append(self.fragment)
        if self.conditions:
            intent=self.conditions.intent(skill,obj)
            if intent:parents.append((intent,'attestation'))
        if skill in ('place','table_place','retreat'):parents.append((self.placement,'precondition'))
        condition_keys=self._skill_condition_keys(skill,obj)
        if self.conditions:
            parents += [(self.conditions.get(key)['id'],'precondition') for key in condition_keys
                        if self.conditions.get(key)['truth']=='TRUE']
        references={}
        if skill=='approach' and sources['grasp']['data'].get('reference_surface'):
            references[obj]=copy.deepcopy(sources['grasp']['data']['reference_surface'])
        elif skill=='place':
            support=sources['support_surface']['data'];references[support['object']]=copy.deepcopy(support)
        return self.graph.add('plan',dict(skill=skill,object=obj,inputs=inputs,params=params or {},
            required_conditions=condition_keys,external_references=references,**target),parents=parents)

    def _skill_condition_keys(self,skill,obj):
        return self.conditions.required(skill,obj) if self.conditions else []

    def _effects(self,name):
        roots=[]
        changed_objects={self.held_object} if (self.closed or name=='open_gripper') and self.held_object else set()
        active=self.graph.nodes.get(self.active_plan,{}).get('data',{}) or {}
        if active.get('skill')=='place':
            sid=active['inputs']['support_surface']
            changed_objects.add(self.graph.nodes[sid]['data'].get('object'))
        for n in self.graph.nodes.values():
            if not n['valid'] or n['kind']!='estimate':continue
            d=n['data'] or {}
            if d.get('type')=='robot' or (d.get('object') in changed_objects and d.get('type') in ('object','surface','grasp')):
                roots.append(n['id'])
        if not self.graph_options['selective_invalidation']:
            roots += [n['id'] for n in self.graph.nodes.values() if n['kind'] in ('plan','check') and n['valid']]
        if roots:self.graph.invalidate(roots,'returned_motion_changed_public_state')
        for obj in changed_objects:
            if obj in self.object_epochs:
                self.object_epochs[obj]+=1
                if self.active_dependencies is not None and obj in self.active_dependencies['external_epochs']:
                    # Declare only this expected effect. capture() separately compares
                    # public geometry to history even across this epoch boundary.
                    self.active_dependencies['external_epochs'][obj]+=1
        self.graph._event('physical_effect',api=name,phase=self.phase,held_object=self.held_object,
                          object_epochs=self.object_epochs,invalidated_sources=roots)
        if self.memory and name in ('open_gripper','close_gripper'):
            for obj in self.object_epochs:self.memory.invalidate(obj,'touch_or_grip_transition')
        if self.conditions:self.conditions.motion_effect(self,name)

    def _grasp_lost(self,reason):
        """Commit existing public empty-jaw evidence; keep actions as history."""
        obj=self.held_object
        if self.memory:self.memory.invalidate(obj,'grasp_lost')
        parents=[n for n in (self.last_observation,self.last_call,self.attachment_id) if n]
        roots=[n['id'] for n in self.graph.nodes.values() if n['valid']
            and n['kind'] in ('estimate','plan') and (n.get('data') or {}).get('object')==obj]
        roots += [self.placement]
        self.graph.invalidate(roots,'public_empty_jaws_end_grasp',conservative=not self.graph_options['selective_invalidation'])
        fact=self.graph.add('verification_fact',{'type':'public_grasp_loss','historical_only':True,
            'object':obj,'grasp_epoch':self.grasp_epoch,'reason':reason,
            'normalized_jaw_opening':float(self.last_public['robot_cartesian_pos'][7]),
            'object_location_known':False,'completed_actions_remain_history':True},parents=parents)
        self.last_grasp_loss={'object':obj,'reason':reason,'grasp_epoch':self.grasp_epoch,'source_id':fact}
        if self.conditions:
            for predicate in ('holding','lifted'):
                self.conditions.set(predicate+':'+obj,'FALSE',evidence=[fact],reason='public_grasp_lost')
        if obj in self.object_epochs:self.object_epochs[obj]+=1
        self.phase='ready';self.closed=self.lifted=False;self.held_object=None
        self.approached_object=None;self.attachment=None;self.attachment_id=None
        self.last_place=None;self.baseline=None;self.close_tcp_z=None
        self.released=False;self.retreat_completed=False
        # Counters, call history and historical graph facts deliberately persist.
        self.record('L2','grasp_loss','FAIL',reason,self.last_grasp_loss,
            action='regrasp',graph_parents=[fact])
        self._stop(reason,[fact,self.last_check],metrics={'recoverable':True,
            'next_action':'open_empty_gripper_then_fresh_observation_and_regrasp'})

    def _move(self,p,q,z,fn,kind='motion'):
        before=self._tracked(fn)('goto_pose',[p,q],{'z_approach':z})
        self._effects('goto_pose');state,_=self.capture()
        self._view_after_motion(state,kind)
        status,metrics=self.control_check([p,q],{},state)
        self.record('L1','tcp_tracking',status,'tcp_tracking',metrics)
        if status!='PASS':
            if self.local_attempts>=self.max_episode_local:
                self.terminal_reason='local_recovery_budget_exhausted';self._stop('control_retry_budget_exhausted')
            self.local_attempts+=1;self.recoveries+=1
            self._tracked(fn)('goto_pose',[p,q],{'z_approach':0.});self._effects('goto_pose')
            state,_=self.capture();self._view_after_motion(state,kind);status,metrics=self.control_check([p,q],{},state)
            self.record('L1','tcp_tracking_retry',status,'tcp_tracking',metrics)
            if status!='PASS':self._stop('control_not_converged')
        if self.closed and float(state['robot_cartesian_pos'][7])<.05:
            self._grasp_lost('lost_grasp_empty_jaws')
        return before

    def _precise_place(self,p,q,fn):
        def reserve():
            if self.local_attempts>=self.max_episode_local:
                self.terminal_reason='local_recovery_budget_exhausted';return False
            self.local_attempts+=1;self.recoveries+=1
            return True
        def execute(name,args,kwargs):
            if self.active_plan:
                plan=self.graph.nodes[self.active_plan]['data']
                if plan['skill']=='table_place':self._current_table_check(plan['inputs']['table_region'])
            self._tracked(fn)(name,args,kwargs);self._effects(name);state,_=self.capture()
            self._view_after_motion(state,'precise_place')
            if self.last_public['robot_cartesian_pos'][7]<.05:
                self._grasp_lost('slipped_or_empty_during_precise_place')
        def emit(event):
            if self.active_dependencies is not None and event['check']=='precise_tcp_command':
                self.active_dependencies['stage']=event['stage']
            self.record('L1',event['check'],event['status'],event['reason'],
                {k:v for k,v in event.items() if k not in ('schema','layer','check','status','reason')})
            if event['check']=='precise_tcp_pre_descent' and self.active_plan:
                plan=self.graph.nodes[self.active_plan]['data']
                self._prepare_descent(p,q)
                if plan['skill']=='table_place' and not (self.evidence_planning and self.memory):self._current_table_check(plan['inputs']['table_region'])
        result=precise_goto(p,q,observe=lambda:self.capture()[0],execute=execute,
            reserve_correction=reserve,emit=emit,lift_verified=self.phase=='holding_verified',
            max_corrections=self.max_local)
        self.recovery_outcomes.append({'kind':'precision_control','corrections':result['corrections'],'success':True})
        return result

    def _lift_check(self):
        state,_=self.capture();surface=self._surface(self.held_object)
        rise=float(surface['hi'][2]-self.baseline['top_z']) if surface and self.baseline else None
        good=rise is not None and rise>.035
        self.record('L2','lift','PASS' if good else 'UNCERTAIN','public_pregrasp_baseline_rise',
                    {'object':self.held_object,'object_rise_m':rise,'baseline_available':self.baseline is not None})
        self.lifted=good;self.phase='holding_verified' if good else 'holding_unverified'
        if not good:self._stop('verify_hold_required_missing_or_occluded_baseline',metrics={'recoverable':self.hold_probes<2})

    def _verify_hold(self,fn):
        if not self.closed:self._stop('no_grasp_to_verify')
        if self.lift_attempts<1:self._stop('lift_required_before_hold_probe')
        if self.hold_probes>=2:
            self.terminal_reason='hold_evidence_unrecoverable';self._stop('hold_evidence_unrecoverable',metrics={'recoverable':False})
        self.hold_probes+=1
        self.phase='holding_unverified';self.lifted=False
        state,_=self.capture();s=self._surface(self.held_object)
        tcp=np.asarray(state['robot_cartesian_pos']).copy()
        if self.close_tcp_z is None or tcp[2]-self.close_tcp_z<.05:
            self._stop('lift_clearance_unverified',metrics={'recoverable':False})
        if not s:
            # Acquire a view under the SAME public upright-attachment/height/
            # neighbour constraints. This never promotes an unverified grasp.
            # A separate fresh co-motion probe below must still pass.
            self._recover_held_visibility(fn,single_step=True,verification_probe=True)
            state,_=self.capture();s=self._surface(self.held_object)
            tcp=np.asarray(state['robot_cartesian_pos']).copy()
        if not s:
            if self.hold_probes>=2:self.terminal_reason='hold_evidence_unrecoverable'
            self._stop('hold_surface_occluded',metrics={'recoverable':self.hold_probes<2 and self.terminal_reason is None})
        before=copy.deepcopy(s)
        self._phase_guard('verify_hold',True,'current_hold_probe_preconditions_verified')
        p=tcp[:3].copy();p[0]+=.02
        self._move(p.tolist(),tcp[3:7].tolist(),0.,fn,'hold_probe')
        self.capture();after=copy.deepcopy(self._surface(self.held_object))
        self._move(tcp[:3].tolist(),tcp[3:7].tolist(),0.,fn,'hold_probe_return')
        delta=(np.asarray(after['center'])-np.asarray(before['center'])) if after else None
        good=delta is not None and np.linalg.norm(delta-[.02,0,0])<=.012
        self.record('L2','hold_probe','PASS' if good else 'UNCERTAIN','public_visual_comotion',
                    {'observed_delta':None if delta is None else delta.tolist(),'commanded_delta':[.02,0,0],
                     'contact_sensor':'unavailable','same_camera_limitation':True})
        self.recovery_outcomes.append({'kind':'hold_probe','success':bool(good),'public_only':True})
        if not good:
            if self.hold_probes>=2:self.terminal_reason='hold_evidence_unrecoverable'
            self._stop('hold_probe_inconclusive',metrics={'recoverable':self.hold_probes<2})
        self.phase='holding_verified';self.lifted=True
        return {'status':'PASS','evidence_id':self.last_check}

    def _run(self,plan_id,fn):
        history=self.graph.nodes.get(plan_id)
        if history and history['kind']=='plan':self.skill_intent=copy.deepcopy(history['data'])
        node=self._valid(plan_id)
        if node['kind']!='plan' or plan_id in self.consumed:self._stop('completed_or_nonplan_cannot_execute',[plan_id])
        d=node['data'];skill=d['skill']
        for source in d['inputs'].values():self._valid(source)
        reason=None
        if self.phase=='released' and skill!='retreat':reason='released_requires_retreat'
        elif skill=='approach' and self.closed:reason='already_holding_no_approach'
        elif skill=='lift' and (not self.closed or self.phase=='holding_unverified' and self.lift_attempts>=1):
            reason='verify_hold_before_another_lift'
        elif skill in ('place','table_place') and self.phase!='holding_verified':reason='verify_hold_before_place'
        elif skill=='retreat' and self.phase!='released':reason='retreat_requires_release'
        self._phase_guard(skill,reason is None,reason or 'current_skill_phase_verified',[plan_id])
        if self.conditions:
            missing=[key for key in self._skill_condition_keys(skill,d['object'])
                     if self.conditions.get(key)['truth']!='TRUE']
            if missing:self._stop('task_condition_requires_revalidation',[plan_id],metrics={
                'failed_conditions':missing,'recoverable':True,'next_action':'verify_task_or_restore_required_condition'})
        if skill=='place' and self.fault=='support_refresh' and not self.fault_fired:
            self.fault_fired=True;name=self.graph.nodes[d['inputs']['support_surface']]['data']['object']
            self._surfaces(name);self._valid(plan_id)
        if skill=='table_place':self._current_table_check(d['inputs']['table_region'])
        if skill=='approach':
            self.capture();s=self._surface(d['object']);tcp=self.last_public['robot_cartesian_pos']
            fingers=np.asarray(self.last_public.get('robot_gripper_joint_pos',[]),float)
            if fingers.shape!=(2,) or not np.isfinite(fingers).all():
                self._stop('finger_encoder_evidence_required',[plan_id],metrics={'recoverable':False})
            fingers=np.abs(fingers)
            if np.min(fingers)<.035 or np.max(fingers)-np.min(fingers)>.002:
                self._stop('open_gripper_before_approach',[plan_id],metrics={'recoverable':True,'next_action':'open_gripper_then_fresh_robot_and_plan'})
            neighbors=[name for name in self.object_epochs if name!=d['object']]
            sweep=panda_open_sweep(d['position'],d['quaternion_wxyz'],
                {name:self.colored_points.get(name,np.empty((0,3))) for name in neighbors},
                opening_m=float(np.clip(fingers.max()*2,0,.08)),z_approach=d['z_approach'])
            self.record('L0','gripper_sweep',sweep['status'],sweep['reason'],sweep,
                action='approach' if sweep['status']=='PASS' else 'reobserve',graph_parents=[plan_id])
            if sweep['status']!='PASS':self._stop('gripper_sweep_'+sweep['reason'],[plan_id,self.last_check],metrics={'recoverable':'fresh_perception_or_different_grasp'})
            sweep_gate=self.last_check;self._valid(sweep_gate)
            self.active_authorization=self.graph.add('verification_fact',{
                'historical_only':True,'type':'boundary_sweep_authorization',
                'source_check_id':sweep_gate,'plan_id':plan_id,'block':self.block},
                parents=[self.active_authorization,(sweep_gate,'precondition')])
            self.baseline=dict(top_z=float(s['hi'][2]),tcp_z=tcp[2],surface=copy.deepcopy(s)) if s else None
            if self.baseline:
                self.baseline['source_id']=self.graph.add('verification_fact',{
                    'type':'public_pregrasp_baseline','historical_only':True,
                    'object':d['object'],**self.baseline},parents=[self.last_observation])
            self.phase='approaching';self.approached_object=None;self.attachment=None;self.attachment_id=None
            self.retreat_completed=False;self.released=False;self.skill_facts.clear()
        external={}
        if skill=='approach':external[d['object']]=[d['inputs']['grasp']]
        if skill=='place':
            sid=d['inputs']['support_surface'];name=self.graph.nodes[sid]['data']['object']
            external[name]=[sid]
        self.active_dependencies={'sources':list(d['inputs'].values()),'started':False,
            'external_sources':external,'external_epochs':{name:self.object_epochs[name] for name in external},
            'external_references':copy.deepcopy(d.get('external_references',{}))}
        self.consumed.add(plan_id);self.graph.consume_plan(plan_id);self.active_plan=plan_id
        if skill=='lift':self.lift_attempts+=1  # Reserve before possibly partial physical effect; never refund.
        try:
            p=list(d['position']);q=list(d['quaternion_wxyz'])
            if skill=='place' and self.fault=='target_shift' and not self.fault_fired:
                # A declared stress perturbation, identical for G0/G1; never truth-based.
                p[0]+=.025;self.fault_fired=True
                self.graph._event('stress_target_shift',plan_id=plan_id,delta=[.025,0,0])
            if skill in ('place','table_place'):self._precise_place(p,q,fn)
            else:self._move(p,q,d['z_approach'],fn)
            if skill=='approach':
                self.phase='approached';self.approached_object=d['object']
                if self.approached_object in self.geometry:
                    geom=self.geometry[self.approached_object];geom['orientation_verified']=True
                    geom['orientation_evidence']='upright_approach_and_rigid_attachment_assumption_not_contact_sensor'
                    previous=self.geometry_ids[self.approached_object]
                    self.geometry_ids[self.approached_object]=self.graph.add('verification_fact',{'type':'conditional_upright_grasp',**geom},parents=[previous,self.last_call,self.last_observation])
            elif skill=='lift':
                self._lift_check()
            elif skill in ('place','table_place'):
                self.last_place=copy.deepcopy(d)
            elif skill=='retreat':
                self.phase='ready';self.retreat_completed=True;self.released=False;self.approached_object=None;self.held_object=None
            self.graph._event('skill_returned',plan_id=plan_id,skill=skill,call_id=self.last_call,phase=self.phase)
            if self.conditions:
                obj=d.get('object') or (self.last_place or {}).get('object')
                dest=next((dst for name,dst in self.conditions.stages if name==obj),None)
                fact=self.conditions.returned(skill,obj,dest,self.last_call,plan_id)
                if skill=='retreat':
                    self.conditions.set('safe_retreat','TRUE',evidence=[fact],reason='public_tcp_verified_retreat')
                self.conditions.refresh(self)
            return {'plan_id':plan_id,'status':'returned','call_id':self.last_call}
        except ExecutionInterrupted as exc:
            if (skill=='table_place' and str(exc)=='pre_motion_rejected'
                and (self.inflight or {}).get('status')=='not_executed'):
                region_id=d['inputs']['table_region'];region=copy.deepcopy(self.graph.nodes[region_id]['data'])
                self.graph.invalidate([region_id],'table_target_ik_rejected',
                    conservative=not self.graph_options['selective_invalidation'])
                parents=[region_id,plan_id]+([self.inflight['id']] if self.inflight.get('id') else [])
                failure=self.graph.add('verification_fact',{'type':'rejected_table_target',
                    'historical_only':True,'region_id':region_id,'plan_id':plan_id,
                    'object':self.held_object,'grasp_epoch':self.grasp_epoch,
                    'physical_call_executed':False,'scope':'current_grasp_search_exclusion_not_global_unreachability'},parents=parents)
                self.rejected_table_targets.append({'region_id':region_id,'source_id':failure,
                    'object':self.held_object,'grasp_epoch':self.grasp_epoch,'region':region})
                self.last_place=None
                self._stop('pre_motion_table_target_rejected',[plan_id,failure],
                    metrics={'recoverable':True,'rejected_region_id':region_id,
                        'physical_call_executed':False,
                        'next_action':'observe_alternative_table_region_then_refresh_surface_robot_and_plan'})
            raise
        finally:self.active_plan=None;self.active_dependencies=None

    def _release_evidence(self,target,held=None):
        if self.conditions:
            required=self.conditions.required(target['skill'],self.held_object)
            missing=[key for key in required if self.conditions.get(key)['truth']!='TRUE']
            if missing:self._stop('task_condition_requires_revalidation_before_release',
                [self.conditions.get(key)['id'] for key in missing if self.conditions.get(key)['id']],
                metrics={'failed_conditions':missing,'recoverable':True,
                         'next_action':'verify_or_restore_prior_task_stage_before_release'})
        table_gate=None
        if target['skill']=='table_place':
            # Table validation captures again. All held/release evidence must
            # use that newer frame, never a held estimate from before it.
            self._current_table_check(target['inputs']['table_region'])
            table_gate=self.last_check;held=self._held_surface()
        if held is None:held=self._held_surface()
        held_parents=[self.last_observation]
        if self.geometry_ids.get(self.held_object):held_parents.append(self.geometry_ids[self.held_object])
        if self.attachment_id:held_parents.append(self.attachment_id)
        if held and held.get('memory_id'):held_parents.append(held['memory_id'])
        if held and held.get('attachment_check_id'):held_parents.append(held['attachment_check_id'])
        elif self.attachment is not None and self.last_check:held_parents.append(self.last_check)
        # These are THIS release attempt's public estimates, not the old
        # target inputs whose poses were invalidated by the place motion.
        held_evidence=self.graph.add('estimate',{'type':'surface','object':self.held_object,
            'evidence_role':'pre_release_held','surface':copy.deepcopy(held),
            'object_epoch':self.object_epochs[self.held_object]},parents=list(dict.fromkeys(held_parents)))
        if target['skill']=='table_place':
            region=self.graph.nodes[target['inputs']['table_region']]['data'];support={'center':region['center'],'hi':[0,0,region['top_z']]}
            support_name='observed_table_region'
            support_parents=[self.last_observation,target['inputs']['table_region'],table_gate]
        else:
            support_name=self.graph.nodes[target['inputs']['support_surface']]['data']['object'];support=self._current_state_surface(support_name)
            support_parents=[self.last_observation]
            if self.geometry_ids.get(support_name):support_parents.append(self.geometry_ids[support_name])
        predicted=any((s or {}).get('evidence_kind')=='predicted' for s in (held,support))
        if predicted:
            if support and support.get('evidence_kind')!='predicted':
                support=self._calibrated_direct_surface(support_name)
            if held and held.get('evidence_kind')!='predicted':
                held=self._calibrated_direct_surface(self.held_object)
            # Rebind the actual estimates used by the interval checker. The
            # earlier diagnostic estimate is not evidence for another method.
            held_evidence=self.graph.add('estimate',{'type':'bounded_release_held','surface':copy.deepcopy(held)},
                parents=held_parents+([held['memory_id']] if (held or {}).get('memory_id') else [])
                    +([held['calibration_model_id']] if (held or {}).get('calibration_model_id') else []))
        if (support or {}).get('calibration_model_id'):support_parents.append(support['calibration_model_id'])
        support_evidence=self.graph.add('estimate',{'type':'surface','object':support_name,
            'evidence_role':'pre_release_support','surface':copy.deepcopy(support),
            'current_observation_only':not (support or {}).get('evidence_kind')=='predicted'},parents=list(dict.fromkeys(support_parents+([(support or {})['memory_id']] if (support or {}).get('memory_id') else []))))
        check=place_metrics(held,support,self.geometry.get(self.held_object))
        if predicted:
            from .state_memory import bounded_place_metrics
            bounds={k:self.memory.profile[k] for k in ('xy_m','z_m','height_m')}
            check=bounded_place_metrics(held,support,self.geometry.get(self.held_object),
                (held or {}).get('error_bounds',bounds),(support or {}).get('error_bounds',bounds))
        self.record('L2','pre_release',check['status'],check['reason'],check,
            action='release' if check['status']=='PASS' else 'align',graph_parents=[held_evidence,support_evidence])
        return check

    def _release(self,fn):
        reason=('lift_task_requires_keep_holding' if self.task_definition_destination(self.held_object)=='held' else
                'verified_place_required_before_release' if self.phase!='holding_verified' or self.last_place is None else None)
        self._phase_guard('release',reason is None,reason or 'verified_place_phase_allows_release_check')
        target=self.last_place
        for attempt in range(self.max_local+1):
            self.capture();held=self._held_surface()
            check=self._release_evidence(target,held)
            if check['status']=='PASS':
                release_gate=self.last_check;self._valid(release_gate)
                self.release_permission=release_gate
                self.active_authorization=self.graph.add('verification_fact',{
                    'historical_only':True,'type':'boundary_release_authorization',
                    'source_check_id':release_gate,'block':self.block},
                    parents=[self.active_authorization,(release_gate,'precondition')])
                break
            if check['status']=='UNCERTAIN' or attempt>=self.max_local or self.local_attempts>=self.max_episode_local:
                if self.local_attempts>=self.max_episode_local:self.terminal_reason='local_recovery_budget_exhausted'
                self._stop('pre_release_'+check['reason'],metrics={'recoverable':check['status']!='UNCERTAIN' and self.local_attempts<self.max_episode_local})
            tcp=np.asarray(self.last_public['robot_cartesian_pos']);delta=np.r_[check['delta_xy_m'],.006-check['gap_m']]
            n=np.linalg.norm(delta[:2]);delta[:2]*=min(1.,.035/max(n,1e-8));delta[2]=np.clip(delta[2],-.04,.02)
            self.local_attempts+=1;self.recoveries+=1
            self._move((tcp[:3]+delta).tolist(),tcp[3:7].tolist(),0.,fn,'align')
            old=self.placement
            self.graph.invalidate([old],'local_alignment_changed_placement',conservative=not self.graph_options['selective_invalidation'])
            self.placement=self.graph.version('placement_context','estimate',{'reason':'local_alignment'},parents=[self.last_call],invalidate=False)
            self.recovery_outcomes.append({'kind':'alignment','attempt':attempt,'before':check})
        self._tracked(fn)('open_gripper',[],{})
        self.closed=self.lifted=False;self.phase='released';self.released=True;self.retreat_completed=False
        self._effects('open_gripper')
        if self.conditions:
            dest=next((dst for obj,dst in self.conditions.stages if obj==self.held_object),None)
            self.conditions.returned('release',self.held_object,dest,self.last_call)
        self.record('L2','release','PASS','public_geometry_allowed_release',{'physical_success':'independent_evaluator_only'})
        return None

    def _dispatch(self,name,args,kwargs,fn):
        self._fragment(name)
        if name=='observe_object':return self._observe_object(*args,**kwargs,fn=self._tracked(fn))
        if name=='observe_grasp':return self._grasp(*args,**kwargs,fn=self._tracked(fn))
        if name=='observe_surfaces':return self._surfaces(*args,**kwargs,fn=fn)
        if name=='reobserve_table':return self._reobserve_table(*args,**kwargs,fn=fn)
        if name=='observe_table':
            obj=args[0] if args else kwargs['name']
            self._object_binding(obj)
            self.capture()
            surfaces=self.last_evidence;shape_source=None
            if self.closed and obj==self.held_object:
                # Raw colour extrema can include reflected/antialiased pixels.
                # Reuse the existing CURRENT attachment/visible-face verifier;
                # a prior size alone or raw fallback is not a valid held shape.
                held=self._held_surface();geometry_id=self.geometry_ids.get(obj)
                if held is None or geometry_id is None:
                    self._stop('current_held_shape_required_for_table_search',metrics={
                        'object':obj,'recoverable':self.terminal_reason is None and self._view_budget()['change_view_available'],
                        'next_action':'observe_held_surface_before_table_search_or_stop',**self._view_budget()})
                surfaces={**surfaces,self._object_binding(obj)['sensor_key']:held}
                # This is the historical geometry used to compute candidate
                # bounds, not a current pose handle or release authorization.
                # Refreshing the held pose must not erase a fixed table intent.
                # Actual placement and release still call _table_evaluation.
                shape_source=self.graph.add('verification_fact',{
                    'type':'table_search_held_shape','historical_only':True,'permission':False,
                    'object':obj,'grasp_epoch':self.grasp_epoch,'shape':copy.deepcopy(held),
                    'fit_check_id':held.get('attachment_check_id'),
                    'scope':'candidate_search_footprint_and_anchor_only_not_motion_clearance'},
                    parents=[self.last_observation,geometry_id]+([self.attachment_id] if self.attachment_id else []))
            rejected=[r for r in self.rejected_table_targets
                if r['object']==obj and r['grasp_epoch']==self.grasp_epoch]
            exclusions=[dict(r['region'],id=r['region_id']) for r in rejected]
            options={'excluded_regions':exclusions} if exclusions else {}
            region=choose_table_region(self.points,self.table_z,surfaces,self._object_binding(obj)['perception_name'],**options)
            if region is None:self._stop('no_observed_clear_table_region',metrics={'recoverable':'reperceive'})
            if shape_source:region['search_shape_source_id']=shape_source
            return self._new('table_region',dict(type='table_region',object=obj,**region),
                [r['source_id'] for r in rejected]+([shape_source] if shape_source else []))
        if name=='read_robot':return self._robot()
        if name=='plan_skill':return self._plan_skill(*args,**kwargs)
        if name=='run_plan':return self._run(*args,**kwargs,fn=fn)
        if name=='verify_hold':return self._verify_hold(fn)
        if name=='prepare_evidence':return self._prepare_evidence(fn)
        if name=='declare_intent':
            if not self.conditions:self._stop('task_conditions_not_enabled')
            return self.conditions.declare(args[0] if args else kwargs['rows'],self.program)
        if name=='verify_task':
            if not self.conditions:self._stop('task_conditions_not_enabled')
            self.capture()
            return self.conditions.summary()
        if name=='recall_result':
            value=self._valid(args[0] if args else kwargs['id'])
            if value['kind']!='estimate':self._stop('only_estimates_recallable')
            return {'id':value['id'],**copy.deepcopy(value['data'])}
        if name in ('goto_pose','plan_motion','get_object_pose','sample_grasp_pose'):
            self._stop('use_grounded_skill_and_public_source_contract',incomplete=True)
        if self.phase=='released' and name in ('open_gripper','close_gripper','home_pose'):
            self._phase_guard(name,False,'released_requires_retreat')
        if name=='close_gripper':
            reason=('already_holding_do_not_reset_grasp' if self.closed else
                    'approach_required_before_close' if self.phase!='approached' else None)
            self._phase_guard(name,reason is None,reason or 'returned_approach_allows_close')
            self._tracked(fn)(name,args,kwargs)
            # Commit the returned command before optional RGB-D parsing. A
            # missing camera frame must not authorize another close/home action.
            # This is commanded closure, not proof of grasp or task success.
            self.closed=True;self.lifted=False;self.held_object=self.approached_object;self.phase='holding_unverified'
            self.lift_attempts=0;self.hold_probes=0;self.last_place=None;self.grasp_epoch+=1
            self.visibility_reference_q=None;self.visibility_context=None  # New grasp, persistent episode budgets.
            self.retreat_completed=False;self.released=False
            self.close_tcp_z=float(self.last_public['robot_cartesian_pos'][2])
            self.attachment=None;self.attachment_id=None
            self._effects(name)
            self.capture()
            if self.last_public['robot_cartesian_pos'][7]<.05:
                self._grasp_lost('empty_grasp')
            self.attachment=calibrate_attachment((self.baseline or {}).get('surface'),self.geometry.get(self.held_object),
                self.last_public['robot_cartesian_pos'],source_id=(self.baseline or {}).get('source_id'),tcp_id=self.last_observation)
            if self.attachment:
                self.attachment_id=self.graph.add('verification_fact',{'type':'public_rigid_grasp_attachment',**self.attachment},parents=self.attachment['sources']+[self.last_call])
            if self.fault=='missing_baseline' and not self.fault_fired:self.baseline=None;self.fault_fired=True
            self.record('L2','grasp','UNCERTAIN','jaw_obstruction_requires_lift_or_probe',{'object':self.held_object,'baseline_available':self.baseline is not None})
            if self.conditions:
                self.conditions.returned('close_gripper',self.held_object,
                    self.conditions.destination(self.held_object),self.last_call)
            return None
        if name=='open_gripper':
            if self.closed:return self._release(fn)
            self._phase_guard(name,True,'empty_gripper_open_allowed')
            self._tracked(fn)(name,args,kwargs);self._effects(name);return None
        if name=='home_pose':
            self._phase_guard(name,not self.closed,'home_while_holding_forbidden' if self.closed else 'empty_home_allowed')
            self.phase='ready';self.approached_object=None;self.baseline=None
            result=self._tracked(fn)(name,args,kwargs);self._effects(name);return result
        self._stop('unknown_grounded_api',incomplete=True)

    def finish_check(self):
        self.capture()
        from .task_definition import resolve_definition
        definition=getattr(self,'task_definition',None) or resolve_definition(self.task)
        holding_goal=any(k.startswith('holding:') for k in definition['goals'])
        candidate=(holding_goal and self.phase=='holding_verified' or
                   not holding_goal and self.phase=='ready' and self.retreat_completed and not self.closed and not self.released)
        conditions=getattr(self,'conditions',None)
        if conditions:candidate=candidate and conditions.complete()
        return self.record('L2','task_candidate','PASS' if candidate else 'UNCERTAIN',
            'public_task_conditions_not_physical_success' if conditions else 'public_stage_only_not_physical_success',
            {'phase':self.phase,**({'remaining_goals':conditions.remaining_goals()} if conditions else {})},
            action='finish_candidate' if candidate else 'replan')

    def summary(self):
        value=super().summary()
        if self.memory:
            value['state_memory']={obj:dict(status=r.get('current_status',r['status']),source_id=r['node'],
                evidence_ref=r['evidence_ref'],reason=r.get('invalid_reason'),
                attached=r['attached'],travel_m=r['path_m'],age_seconds=max(0,time.monotonic()-r['stamp']),
                current_visual_detection=r.get('current_status')=='DIRECT',model_id=self.memory.profile['calibration_id'])
                for obj,r in self.memory.records.items()}
        last_guard=next((e for e in reversed(self.events) if e['check']=='dependency_guard'),None)
        allowed={'recoverable','next_action','missing_geometry_objects','reobserve_change_view','region_id','program_error','rejected_region_id','missing_surface_object','perception_failure_behavior','failed_conditions'}
        value['recovery_preconditions']={k:copy.deepcopy(v) for k,v in
            (last_guard['metrics'] if last_guard else {}).items() if k in allowed}
        if not self.graph_options['diagnostic_slice']:
            value['diagnostic_source']=self.source
            value['diagnosis'].update(schema='capx.flat_diagnostic.v2',nodes=list(self.graph.nodes.values())[-24:])
        if not self.graph_options['selective_recovery']:
            scope={'policy':'all_unexecuted_suffix','never_replay':list(self.call_history),'reason':'conservative_recovery_policy'}
            value['recovery_scope']=scope;value['diagnosis']['recovery_scope']=scope
        value['graph_options']=dict(self.graph_options)
        value.update(schema='capx.grounded_feedback.v2',task=self.task,phase=self.phase,
            held_object=self.held_object if self.closed else None,grasp_active=bool(self.closed),lift_verified=bool(self.lifted),
            last_manipulated_object=self.held_object or (self.last_grasp_loss or {}).get('object'),
            last_grasp_loss=copy.deepcopy(self.last_grasp_loss),
            baseline_available=self.baseline is not None,lift_attempts=self.lift_attempts,hold_probes=self.hold_probes,
            visibility_probes=self.visibility_probes,local_attempts=self.local_attempts,perception_refinements=self.perception_refinements,**self._view_budget(),
            terminal_reason=self.terminal_reason,reconciliation_required=copy.deepcopy(self.reconciliation_required),recovery_outcomes=self.recovery_outcomes[-6:],
            recovery_instruction=('verify_hold if unverified; never reset an active close stage'
                if self.closed else 'After a verified release, fresh retreat only'
                if self.phase=='released' else 'No active grasp. Open empty jaws before a new approach; observe the current scene and replan remaining work. Historical object identity is not a held object or permission to place.'))
        # All public source roles, not just v1 robot/grasp estimates.
        value['available_results']=[{'id':n['id'],**copy.deepcopy(n['data'])} for n in self.graph.nodes.values()
            if n['kind']=='estimate' and n['valid'] and isinstance(n['data'],dict) and n['data'].get('type') in ('object','surface','grasp','robot','table_region')][-16:]
        if self.conditions:
            if self.closed and self.phase=='holding_verified' and self.last_place:
                obj=self.held_object;dest=next((d for o,d in self.conditions.stages if o==obj),None)
                label=str(obj)+':'+str(dest)
                if 'positioned:'+label in self.conditions.history:
                    try:
                        check=self._release_evidence(self.last_place)
                        truth={'PASS':'TRUE','FAIL':'FALSE','UNCERTAIN':'UNKNOWN'}[check['status']]
                        evidence=[self.last_check]
                    except ExecutionInterrupted:
                        truth='UNKNOWN';evidence=[self.last_observation]
                    self.conditions.set('aligned:'+label,truth,evidence=evidence,
                        epochs={'robot':self.robot_epoch,'object':self.object_epochs[obj],
                                'support':self.object_epochs.get(dest)},reason='current_geometry_after_returned_positioning')
            value['task_conditions']=self.conditions.summary()
            value['last_repair_decision']=copy.deepcopy(getattr(self,'last_repair_decision',None))
        # Presentation limits do not change any parent-side admission/recovery
        # decision. All receipts, conditions and graph events remain archived.
        from .diagnostic_context import bounded
        critical={k:value.get(k) for k in ('terminal_reason','phase','reconciliation_required','recovery_preconditions')}
        critical['failed_checks']=[{k:e.get(k) for k in ('layer','check','status','reason')}
            for e in self.events if e.get('status') in ('FAIL','UNCERTAIN')][-4:]
        if self.conditions:
            critical['conditions']=[{'key':k,'truth':self.conditions.get(k)['truth'],
                'reason':self.conditions.get(k)['reason']} for k in self.conditions.ids]
            critical['remaining_goals']=self.conditions.remaining_goals()
        value,stats=bounded(value,max_chars=20000)
        value['critical_current_state']=critical
        value['context_stats']=stats
        return value
