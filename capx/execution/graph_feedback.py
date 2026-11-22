"""Contract-bound online graph recovery, layered over the existing L0-L3 broker.

No simulator imports. IDs refer to parent-owned public evidence. No arbitrary
Python dataflow inference, no mid-motion cancellation, no automatic action replay.
"""
import hashlib
import numpy as np
from .feedback import FeedbackMonitor, public_observation
from .dependency_graph import DependencyGraph
from .sandbox import ExecutionInterrupted, validate

GRAPH_APIS=['observe_object','observe_grasp','read_robot','plan_motion','run_plan','recall_result']
GRAPH_API_DOC='''Additional parent-owned execution contracts (use these in both groups):
observe_object(name): fresh real Vision get_object_pose; returns dict id, position, quaternion, extent. name is red cube or green cube.
observe_grasp(name): real GraspNet sample; returns dict id, position, quaternion.
read_robot(): public robot encoders; returns dict id, tcp (XYZ,WXYZ,gripper fraction), joints.
plan_motion(skill, position, quaternion_wxyz, dependencies, z_approach=0.0): register computed target, returns string plan ID. dependencies is a list of result/plan IDs actually used by your calculation. Skills: approach, lift, place, retreat. place MUST include current red and green observe_object IDs; approach must depend on observe_grasp ID; lift must depend on grasp ID or read_robot ID; retreat must depend on the place plan ID or a fresh read_robot ID. Do not omit dependencies or reuse invalid results.
run_plan(plan_id): execute registered motion ONCE with the existing goto_pose and layered monitors. May interrupt before action if obsolete. Returns receipt, not a new plan.
recall_result(id): retrieve a still-valid public observation/estimate without another service request. Never executes an action. Invalid IDs are refused.
open_gripper(), close_gripper(), home_pose() retain their original meanings. Legacy get_object_pose/sample_grasp_pose are blocked: use observe_object/observe_grasp for version registration. Raw goto_pose is NOT allowed in this experiment (unregistered dependencies). home_pose after release is blocked until a verified retreat has completed.
Register place AND its dependent retreat BEFORE executing place. Then run place, open_gripper, run retreat. Local corrections can invalidate that old retreat. On recovery, use CURRENT public robot pose, register a NEW retreat; do not replay completed grasp/lift/release. First retreat must lift >=0.08m at same XY (<=0.01m lateral) and nearly same orientation (<=0.2rad); this is a boundary waypoint check, not a collision-free path guarantee.
A blocked program terminates at an API boundary. Physical state persists; a fresh actor runs recovery. Feedback identifies completed actions, invalid results, and available valid results. Use retained valid observations when available; if none remain, re-observe required objects. If already released, finish retreat without regrasping. If holding, do not open blindly.
All geometry is robot-base metres; quaternions WXYZ. No simulator truth or tactile contact is available. The runner specifies episode API and code-block budgets; local recovery calls count against them. At most 64 broker calls/block and 90 seconds/block. Return a complete Python block with all variables defined.
'''


class PreMotionRejected(ExecutionInterrupted):
    """Trusted parent callback proved this complete API call never moved.

    Only a trusted preflight proving no IK segment returned within this API
    call may raise this signal. A transport failure or failure after a preceding
    physical segment must retain unknown-effect quarantine.
    This type is not exposed in the actor API.
    """


class GraphFeedbackMonitor(FeedbackMonitor):
    def __init__(self,mode,observe,*,emit=None,graph_emit=None,max_local=2,fault='none',site=None):
        super().__init__('E',observe,emit=emit,max_local=max_local)
        if fault not in ('none','support_refresh'):raise ValueError('Unknown fault')
        self.mode=mode;self.graph=DependencyGraph(mode,graph_emit);self.fault=fault;self.fault_fired=False
        self.site=site or (lambda:None);self.program=None;self.fragment=None;self.active_plan=None
        self.last_call=None;self.last_observation=None;self.roots=[];self.consumed=set()
        self.released=False;self.retreat_completed=False;self.inflight=None;self.call_history=[]
        self.capability=self.graph.add('capability',{'contract':'stack-motion-v1','intervention':'API_boundary_only'})
        self.placement=self.graph.version('placement_context','estimate',{'revision':0},invalidate=False)
        self.block=0;self.robot_epoch=0;self.change_roots=[];self.interruption_roots=[]
        self.conservative_fallback=False;self.fallback_invalidated=[]
        self.skill_facts={};self.close_baseline=None;self.grasp_epoch=0
        self.active_precondition=None;self.skill_failure_evidence=[]
        self.reconciliation_required=None

    def begin_program(self,code,block):
        self.roots=[];self.interruption_roots=[];self.change_roots=[]
        self.conservative_fallback=False;self.fallback_invalidated=[]
        self.skill_failure_evidence=[]
        self.block=block;self.program=self.graph.add('program',{'block':block,'sha256':hashlib.sha256(code.encode()).hexdigest()})
        self.source=code;self.fragment=None
        self.capture()  # Refresh public evidence; this alone never proves an unknown physical effect completed.
        self._reconcile_context()

    def _reconcile_context(self):
        if not self.graph.nodes[self.placement]['valid']:
            self.placement=self.graph.version('placement_context','estimate',
                {'reason':'public_state_reconciliation'},parents=[self.last_observation],invalidate=False)

    def _fragment(self,name):
        site=self.site() or {};line=site.get('line');snippet=site.get('source_line',name+'(...)')
        self.fragment=self.graph.add('fragment',{'block':self.block,'line':line,'source':snippet,'api':name},parents=[self.program] if self.program else [])

    def capture(self):
        state,evidence=super().capture()
        self.last_observation=self.graph.add('observation',{'robot':self.last_public,'surfaces':evidence,'raw_data':'separate observations NPZ'})
        return state,evidence

    def record(self,layer,check,status,reason,metrics=None,action='continue',graph_parents=(),**extra):
        row=super().record(layer,check,status,reason,metrics,action,**extra)
        if hasattr(self,'graph'):
            parents=[p for p in (self.last_call,self.last_observation) if p]+list(graph_parents)
            if layer=='L2' and check=='grasp':
                self.grasp_epoch+=1
                self.close_baseline=self.last_observation
                self.skill_facts.pop('lift',None)
            if layer=='L2' and check=='lift':
                parents += [p for p in (self.close_baseline,self.skill_facts.get('grasp')) if p]
            node=self.graph.add('check',{'layer':layer,'check':check,'status':status,'reason':reason,'metrics':metrics or {}},parents=parents)
            self.last_check=node
            if layer=='L2' and check in ('grasp','lift'):
                # The conclusion and baseline are historical evidence. A new
                # current boundary check below still decides permission.
                fact=self.graph.add('verification_fact',{'historical_only':True,
                    'check':check,'status':status,'reason':reason,'metrics':metrics or {},
                    'source_check_id':node,'grasp_epoch':self.grasp_epoch,
                    'close_baseline_id':self.close_baseline,
                    'baseline_red_visible':bool((self.close_evidence or {}).get('red'))},parents=[node])
                self.skill_facts[check]=fact
            if status!='PASS':self.roots=[node]+([self.active_plan] if self.active_plan else [])
        return row

    def _stop(self,reason,roots=(),incomplete=False,metrics=None):
        self.roots=list(roots) or self.roots
        if incomplete:
            self.conservative_fallback=True
            self.fallback_invalidated.extend(self.graph.invalidate(list(roots),reason,conservative=True))
        self.record('L3','dependency_guard','FAIL',reason,{'mode':self.mode,'inflight':self.inflight,**(metrics or {})},action='replan')
        # Keep the failed plan roots as well as the check for backwards diagnosis.
        self.roots=list(dict.fromkeys(list(roots)+self.roots+self.change_roots))
        raise ExecutionInterrupted(reason)

    def _tracked(self,fn):
        def call(name,args,kwargs):
            parents=[(self.capability,'precondition')]+[(p,'data') for p in (self.active_plan,self.fragment) if p]
            if self.active_precondition:parents.append((self.active_precondition,'precondition'))
            node=self.graph.add('call',{'api':name,'args':args,'kwargs':kwargs,'block':self.block},parents=parents)
            self.inflight={'id':node,'api':name,'status':'started'}
            try:result=fn(name,args,kwargs)
            except PreMotionRejected as exc:
                self.inflight['status']='not_executed'
                self.graph.invalidate([node], 'pre_motion_rejected')
                self._stop('pre_motion_rejected', [node]+([self.active_plan] if self.active_plan else []),
                    metrics={'reason':str(exc),'physical_call_executed':False})
            except BaseException:
                self.inflight['status']='unknown_partial_effects'
                if name in ('goto_pose','open_gripper','close_gripper','home_pose'):
                    self._physical_effect_unknown(name,node)
                raise
            # A physical call that returned is a historical fact even if a later verifier fails.
            self.graph.returned_call(node)
            if name in ('goto_pose','open_gripper','close_gripper','home_pose'):self.robot_epoch+=1
            self.last_call=node;self.inflight={'id':node,'api':name,'status':'returned'};self.call_history.append(node)
            return result
        return call

    def _physical_effect_unknown(self,name,node):
        # A callback can fail after the robot moved. Neither a fresh image nor
        # public encoders prove which path/contact effects completed. Quarantine
        # the unfinished suffix; retain receipts of earlier returned actions.
        self.reconciliation_required={'api':name,'call_id':node,
            'status':'unknown_partial_effects','automatic_resume':False}
        self.robot_epoch+=1
        changed=self.graph.invalidate([node],'unknown_physical_effect',conservative=True)
        self.change_roots=list(dict.fromkeys(self.change_roots+[node]))
        self.graph._event('reconciliation_required',**self.reconciliation_required,
            invalidated=changed,policy='terminal_without_controller_completion_receipt')

    def recovery(self,kind,fn,name,args,kwargs):
        result=super().recovery(kind,fn,name,args,kwargs)
        if kind=='align':
            # The commanded correction can move the carried red object. Its
            # bundled pose/extent estimate must not be offered as unchanged.
            if self.closed:
                red=[n['id'] for n in self.graph.nodes.values() if n['key']=='object:red cube' and n['valid']]
                if red:self.graph.invalidate(red,'held_red_moved_by_local_alignment')
            old=self.placement
            self.placement=self.graph.version('placement_context','estimate',{'reason':'local_alignment','actual_state_must_be_rechecked':True},parents=[self.last_call],reason='local_alignment_changed_placement')
            self.capture()
            # New evidence is attached after invalidation; it must not inherit the old version's invalidity.
            evidence=self.graph.add('check',{'check':'alignment_state_reconciled','public_robot':self.last_public},parents=[self.last_call,self.last_observation,self.placement])
            self.change_roots=[self.placement,evidence]
            self.roots=[old]+([self.active_plan] if self.active_plan else [])
        return result

    def _estimate(self,key,value,parents,force=False):
        previous=self.graph.latest.get(key)
        if previous and self.graph.nodes[previous]['valid']:
            # Routine observation collection must not erase its newly collected sibling inputs.
            # G0 uses a flat all-plan/check suffix, not graph-selected preservation.
            roots=[previous]
            if self.mode=='G0':
                roots += [n['id'] for n in self.graph.nodes.values() if n['kind'] in ('plan','check')]
            self.graph.invalidate(roots,'observation_replaced:'+key,conservative=False)
        # New observation is registered AFTER invalidation so it is usable in both modes.
        state,evidence=self.capture()
        self._reconcile_context()
        node=self.graph.version(key,'estimate',value,parents=[p for p in (self.last_call,self.last_observation) if p],invalidate=False)
        self.change_roots=[node]
        return {'id':node,**value}

    def _observe_object(self,name,fn):
        if name not in ('red cube','green cube'):raise ValueError('Only Stack red/green objects are supported')
        value=super().invoke('get_object_pose',[name],{'return_bbox_extent':True},fn)
        p,q,extent=value
        return self._estimate('object:'+name,dict(object=name,position=np.asarray(p).tolist(),quaternion=np.asarray(q).tolist(),extent=np.asarray(extent).tolist()),[])

    def _valid(self,node):
        try:
            result=self.graph.require_valid(node)
            if result['kind']=='estimate' and result['data'].get('type')=='robot' and (result['data'].get('robot_epoch')!=self.robot_epoch or node!=self.graph.latest.get('robot_read')):
                self.graph.invalidate([node],'robot_state_changed',conservative=self.mode=='G0')
                self._stop('stale_robot_observation',[node])
            return result
        except (KeyError,RuntimeError):self._stop('stale_or_unknown_dependency',[node] if node in self.graph.nodes else [],incomplete=node not in self.graph.nodes)

    def _plan(self,skill,position,quaternion_wxyz,dependencies,z_approach=0.):
        if skill not in ('approach','lift','place','retreat'):raise ValueError('Unknown skill contract')
        p=np.asarray(position,float);q=np.asarray(quaternion_wxyz,float)
        if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all() or abs(np.linalg.norm(q)-1)>.02:raise ValueError('Invalid target geometry')
        if not isinstance(dependencies,(list,tuple)) or not dependencies:self._stop('missing_plan_dependencies',incomplete=True)
        nodes=[self._valid(i) for i in dependencies]
        objects={n['data'].get('object') for n in nodes if isinstance(n['data'],dict)}
        kinds={n['data'].get('type') for n in nodes if isinstance(n['data'],dict)}
        place_parent=any(n['kind']=='plan' and n['data']['skill']=='place' for n in nodes)
        complete=(skill=='place' and {'red cube','green cube'}<=objects or skill=='approach' and 'grasp' in kinds or skill=='lift' and bool(kinds&{'grasp','robot'}) or skill=='retreat' and (place_parent or 'robot' in kinds))
        parents=[(i,'data') for i in dependencies]+[(self.capability,'precondition')]
        if self.fragment:parents.append((self.fragment,'data'))
        if skill in ('place','retreat'):parents.append((self.placement,'precondition'))
        node=self.graph.add('plan',{'skill':skill,'position':p.tolist(),'quaternion':q.tolist(),'z_approach':float(z_approach),'complete_dependencies':bool(complete),'dependencies':list(dependencies)},parents=parents)
        if not complete:self._stop('incomplete_skill_contract',[node],incomplete=True)
        return node

    def _skill_precondition(self,plan_id,skill):
        # Preserve existing permission rules; new graph edges expose WHY a
        # holding/lift precondition is uncertain and link its close baseline.
        self.capture()
        facts=[self.skill_facts[key] for key in ('grasp','lift')
               if key in self.skill_facts and (key=='grasp' or skill=='place')]
        allowed=self.closed and (skill!='place' or self.lifted)
        reason=('holding_precondition_missing' if not self.closed else
                'lift_precondition_unverified' if skill=='place' and not self.lifted else
                'current_skill_preconditions_verified')
        lift_fact=self.graph.nodes.get(self.skill_facts.get('lift'),{})
        status='PASS' if allowed else ('UNCERTAIN' if self.closed and
            lift_fact.get('data',{}).get('status')=='UNCERTAIN' else 'FAIL')
        self.record('L2','skill_precondition',status,reason,
            {'skill':skill,'gripper_commanded_closed':self.closed,'lift_verified':self.lifted,
             'public_robot':self.last_public,'historical_facts_are_not_permission':True},
            action='continue' if allowed else 'replan',
            graph_parents=[(plan_id,'data')]+[(fact,'precondition') for fact in facts])
        gate=self.last_check
        if not allowed:
            self.skill_failure_evidence=facts
            self._stop(reason,[plan_id,gate]+facts)
        return gate

    def _run(self,plan_id,fn):
        node=self._valid(plan_id)
        if node['kind']!='plan' or plan_id in self.consumed:self._stop('completed_or_nonplan_cannot_execute',[plan_id])
        data=node['data'];skill=data['skill']
        for dependency in data['dependencies']:self._valid(dependency)
        if self.released and not self.retreat_completed and skill!='retreat':
            self._stop('released_object_requires_verified_retreat',[plan_id],incomplete=True)
        # Controlled experiment event: refresh actual support observation, never alter simulator truth.
        if skill=='place' and self.fault=='support_refresh' and not self.fault_fired:
            self.fault_fired=True
            self._observe_object('green cube',fn)
            self._valid(plan_id)  # Must block the now-obsolete target before moving.
        if skill=='approach' and self.closed:self._stop('approach_while_already_holding',[plan_id])
        precondition=self._skill_precondition(plan_id,skill) if skill in ('lift','place') else None
        if skill=='retreat':
            state,_=self.capture();tcp=np.asarray(state['robot_cartesian_pos']);p=np.asarray(data['position']);q=np.asarray(data['quaternion'])
            angle=2*np.arccos(np.clip(abs(np.dot(q,tcp[3:7])),0,1))
            good=self.released and not self.closed and p[2]-tcp[2]>=.08 and np.linalg.norm(p[:2]-tcp[:2])<=.01 and angle<=.2 and data['z_approach']==0
            self.record('L2','retreat_boundary','PASS' if good else 'FAIL','retreat_waypoint_contract',{'delta_xyz':(p-tcp[:3]).tolist(),'angle_rad':float(angle)},action='continue' if good else 'replan')
            if not good:self._stop('retreat_requires_fresh_vertical_plan',[plan_id])
        self.consumed.add(plan_id);self.active_plan=plan_id;self.active_precondition=precondition
        self.graph.consume_plan(plan_id)
        try:
            super().invoke('goto_pose',[data['position'],data['quaternion']],{'z_approach':data['z_approach']},fn)
            if skill=='retreat':self.retreat_completed=True
            return {'plan_id':plan_id,'call_id':self.last_call,'status':'returned'}
        except ExecutionInterrupted:
            self.graph.invalidate([plan_id],'motion_verification_interrupted',conservative=self.mode=='G0')
            self.roots=list(dict.fromkeys([plan_id]+self.roots))
            raise
        finally:self.active_plan=None;self.active_precondition=None

    def invoke(self,name,args,kwargs,fn):
        if self.reconciliation_required:
            self._stop('unknown_physical_effect_terminal_stop',
                [self.reconciliation_required['call_id']],metrics={'recoverable':False,
                    'reconciliation_required':self.reconciliation_required})
        try:
            try:
                return self._dispatch(name,args,kwargs,fn)
            except (ValueError,TypeError) as exc:
                # Callback validation and possibly partial effects are not safe
                # for actor-local try/except continuation. Do not forward raw
                # exception text: a service error may contain request details.
                roots=[p for p in (self.active_plan,self.fragment) if p]
                self._stop('api_contract_error',roots,incomplete=True,
                    metrics={'api':name,'error_type':type(exc).__name__})
        except ExecutionInterrupted:
            self.interruption_roots=list(dict.fromkeys(self.roots+self.change_roots))
            if self.mode=='G0':
                self.graph.invalidate(self.roots,'conservative_exception_suffix',conservative=True)
            raise

    def end_block_failure(self,status,detail=''):
        """Discard an unknown unfinished suffix after an ordinary actor failure.

        Timeout/system errors still use P0 environment rebuild; completed calls
        remain facts. The runner invokes this before preparing recovery input.
        """
        if status not in ('runtime_error','invalid_program','policy_violation'):
            return False
        # Only a controlled class name crosses this boundary, never raw stderr
        # or a transport exception that could include private request details.
        error_type=str(detail).partition(':')[0]
        if error_type not in {'ValueError','TypeError','IndexError','KeyError','NameError',
                'UnboundLocalError','ZeroDivisionError','RuntimeError','AssertionError','SyntaxError'}:
            error_type='ProgramError'
        self.conservative_fallback=True
        self.fallback_invalidated.extend(self.graph.invalidate([],'unclassified_program_failure',conservative=True))
        self.fragment=self.graph.add('fragment',{'block':self.block,'line':None,
            'source':self.source,'api':'program_failure'},parents=[self.program])
        metrics={'execution_status':status,'error_type':error_type,'inflight':self.inflight}
        if status in ('invalid_program','policy_violation'):
            # Re-derive diagnostics from model-owned source with the SAME static
            # validator. Never forward actor stderr or an API exception payload.
            try:validate(self.source)
            except (ValueError,SyntaxError) as exc:
                error={'source':'sandbox_static_validation','error_type':type(exc).__name__,
                       'message':(exc.msg if isinstance(exc,SyntaxError) else str(exc))[:256]}
                if isinstance(exc,SyntaxError):error.update(line=exc.lineno,offset=exc.offset)
                metrics.update(program_error=error,
                    next_action='repair_program_contract_preserve_physical_stage_and_completed_calls')
        self.record('L3','dependency_guard','FAIL','program_runtime_failure',metrics,action='replan')
        self.interruption_roots=list(dict.fromkeys([self.fragment]+self.roots+self.change_roots))
        return True

    def _dispatch(self,name,args,kwargs,fn):
        self._fragment(name);tracked=self._tracked(fn)
        if name=='observe_object':return self._observe_object(*args,**kwargs,fn=tracked)
        if name=='observe_grasp':
            value=super().invoke('sample_grasp_pose',args,kwargs,tracked)
            return self._estimate('grasp',{'type':'grasp','position':np.asarray(value[0]).tolist(),'quaternion':np.asarray(value[1]).tolist()},[])
        if name=='read_robot':
            state,_=self.capture();self._reconcile_context();value={'type':'robot','robot_epoch':self.robot_epoch,'tcp':np.asarray(state['robot_cartesian_pos']).tolist(),'joints':np.asarray(state['robot_joint_pos']).tolist()}
            node=self.graph.version('robot_read','estimate',value,parents=[self.last_observation],invalidate=False)
            return {'id':node,**value}
        if name=='plan_motion':return self._plan(*args,**kwargs)
        if name=='run_plan':return self._run(*args,**kwargs,fn=tracked)
        if name=='recall_result':
            node=self._valid(args[0] if args else kwargs['id'])
            if node['kind']!='estimate':self._stop('only_estimates_can_be_recalled',[node['id']])
            return {'id':node['id'],**node['data']}
        if name in ('get_object_pose','sample_grasp_pose'):self._stop('use_versioned_observation_contract',incomplete=True)
        if name=='goto_pose':self._stop('unregistered_motion_dependencies',incomplete=True)
        if name=='home_pose' and self.closed:self._stop('home_while_holding_requires_motion_contract',incomplete=True)
        if name=='home_pose' and self.released and not self.retreat_completed:self._stop('home_requires_verified_retreat',incomplete=True)
        was_closed=self.closed
        response=super().invoke(name,args,kwargs,tracked)
        if name=='open_gripper' and was_closed:self.released=True;self.retreat_completed=False
        if name=='close_gripper':self.released=False;self.retreat_completed=False
        return response

    def summary(self):
        self.capture()  # Actual state after any returned or partially completed call.
        base=super().summary();base['released']=self.released;base['retreat_completed']=self.retreat_completed
        roots=self.interruption_roots or self.roots or ([self.last_call] if self.last_call else [self.placement])
        if self.mode=='G1':
            if self.conservative_fallback:
                roots=list(dict.fromkeys(roots+self.fallback_invalidated))
            diag=self.graph.diagnose(roots,max_nodes=24)
            fragments=[n['data']['source'] for n in diag['nodes'] if n['kind']=='fragment']
            base['diagnostic_source']=self.source if self.conservative_fallback else '\n'.join(dict.fromkeys(fragments))
            if self.conservative_fallback:
                diag['recovery_scope'].update(policy='all_unexecuted_suffix',fallback=True,
                    fallback_reason='unknown_dependencies_or_program_failure')
        else:
            # Flat conservative baseline: never consult edges for evidence selection or preservation.
            diag={'schema':'capx.conservative_suffix.v1','roots':roots,'nodes':list(self.graph.nodes.values())[-24:],
                  'completed_calls':list(self.call_history),'recovery_scope':{'policy':'all_unexecuted_suffix','never_replay':list(self.call_history)}}
            base['diagnostic_source']=self.source
        available=[{'id':n['id'],**n['data']} for n in self.graph.nodes.values() if n['kind']=='estimate' and n['valid'] and isinstance(n['data'],dict) and (n['data'].get('type')!='robot' or n['data'].get('robot_epoch')==self.robot_epoch and n['id']==self.graph.latest.get('robot_read')) and ('object' in n['data'] or n['data'].get('type') in ('robot','grasp'))]
        changes=[e for e in self.graph.events if e['action'] in ('version','invalidate')]
        base.update(skill_precondition_evidence=[{'id':fact,**self.graph.nodes[fact]['data']} for fact in self.skill_failure_evidence],
                    dependency_changes=changes[-12:],mode=self.mode,diagnosis=diag,recovery_scope=diag['recovery_scope'],available_results=available[-12:],
                    completed_actions=[self.graph.nodes[i]['data']['api'] for i in self.call_history],inflight=self.inflight,
                    limitation='Declared contracts only. Observation validity is conditional; not independent confirmation of geometric truth. No mid-motion cancellation.')
        return base
