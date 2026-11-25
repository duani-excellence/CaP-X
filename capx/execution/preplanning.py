"""Public-only candidate search. Search results are never execution permissions.

Reuses existing IK, frame calibration, shape registration, grasp and placement
geometry. No physical API is invoked. Exact caches include the solver reference
and robot calibration; Skill caches additionally include observed scene/effects.
"""
import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .agent_events import plain,digest
from .bounded_ik import IKConvergenceError
from .region_geometry import rank_placements


class DiagnosticReachability:
    def __init__(self,solver,transform,reference,*,calibration=None,emit=None):
        self.solver=solver;self.diagnostics=solver
        while hasattr(self.diagnostics,'__wrapped__'):self.diagnostics=self.diagnostics.__wrapped__
        self.transform=transform;self.reference=reference
        self.calibration=calibration;self.emit=emit or (lambda *a,**k:None)
        self.cache={};self.records={};self.hits=0

    def query(self,p,q,reference=None):
        began=time.monotonic()
        target=np.asarray(self.transform(np.asarray(p,float),np.asarray(q,float)))
        ref=self.reference() if reference is None else reference
        ref=None if ref is None else np.asarray(ref,float).copy()
        solver_config={k:getattr(self.diagnostics,k,None) for k in ('max_iterations','position_tolerance','orientation_tolerance_deg','calibration_hash')}
        key=digest(plain([target,ref,self.calibration,self.diagnostics.kinematics.urdf_sha256,solver_config]))
        if key in self.cache:
            self.hits+=1;result=dict(copy.deepcopy(self.cache[key]),cache_hit=True,
                original_compute_seconds=self.cache[key]['seconds'],seconds=time.monotonic()-began,new_solver_calls=0)
            self.emit('planning_ik_cache',id=result['id'],status=result['status']);return result
        start=time.monotonic();offset=len(self.diagnostics.events)
        result=dict(id='ik-'+key,status='NO_SOLUTION_FOUND',position=plain(p),quaternion_wxyz=plain(q),
            target_hand_wxyz_xyz=target.tolist(),reference=plain(ref),physical_unreachable=False,
            frame='public_robot_base_TCP_to_calibrated_IK_hand',calibration_hash=self.calibration,
            permission=False,cache_hit=False,scope='kinematics_only_not_collision_or_grasp_certificate',
            solver_configuration=solver_config,
            joint_limits=plain(self.diagnostics.kinematics.limits))
        cacheable=True
        try:
            result['joint_solution']=plain(self.solver(target_pose_wxyz_xyz=target,prev_cfg=ref))
            result['status']='KINEMATIC_SOLUTION'
        except IKConvergenceError as exc:
            result['failure_type']=exc.reason
            cacheable=exc.reason!='ik_solve_budget_exhausted'
        # Provider errors are NOT converted to infeasibility or negatively cached.
        result['iterations']=copy.deepcopy(self.diagnostics.events[offset:])
        result['seconds']=time.monotonic()-start
        result['new_solver_calls']=len(result['iterations'])
        last=result['iterations'][-1] if result['iterations'] else {}
        result['position_error_m']=last.get('position_error_m')
        result['orientation_error_deg']=last.get('orientation_error_deg')
        self.records[result['id']]=result
        if cacheable:self.cache[key]=copy.deepcopy(result)
        self.emit('planning_ik_result',**result);return result

    def __call__(self,p,q):return self.query(p,q)['status']=='KINEMATIC_SOLUTION'


def placement_candidates(held,shape,registration,region,profile,preferred=None,quaternion=None,*,diagnostic_orientations=None,max_positions=12):
    """Original geometric bounds, multiple legal upright orientations.

    Returns full object/TCP mapping, including the measured off-centre attachment.
    Rotations outside the existing upright release capability remain unsupported.
    """
    h=held;original=h['shape'];tcp=np.asarray(h['tcp']);inv=np.linalg.inv(tcp)
    local=inv[:3,:3]@np.asarray(original['center'])+inv[:3,3]
    default=Rotation.from_quat([0,1,0,0]);orientations=[]
    if diagnostic_orientations is not None:orientations=[np.asarray(q,float) for q in diagnostic_orientations]
    elif quaternion is not None:orientations=[np.asarray(quaternion,float)]
    else:
        for yaw in (0,90,-90,180):
            r=Rotation.from_euler('z',yaw,degrees=True)*default
            orientations.append(np.roll(r.as_quat(),1))
    groups=[]
    for q in orientations:
        if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.001:raise ValueError('Unit WXYZ required')
        rot=Rotation.from_quat(np.roll(q,-1)).as_matrix()
        if diagnostic_orientations is None and not np.allclose(rot[:,2],[0,0,-1],atol=1e-6):raise ValueError('Existing release contract supports upright yaw only')
        delta=rot@tcp[:3,:3].T;half=np.abs(delta)@np.r_[original['half_extent_xy'],original['height_m']/2]
        margin=half[:2]+profile['xy_error_m']+registration['residual_95_m']+.025
        lo=np.asarray(region['lo'])+margin;hi=np.asarray(region['hi'])-margin
        if np.any(hi<lo):continue
        xy=[lo+(hi-lo)*[a,b] for a in (.5,.25,.75,0.,1.) for b in (.5,.25,.75,0.,1.)]
        ranked=rank_placements(xy,dict(shape,half_extent_xy=half[:2]),region,preferred=preferred)
        group=[]
        for quality in ranked[:max_positions]:
            center=np.r_[quality['xy'],region['floor_z']+half[2]+.0125]
            p=center-rot@local
            group.append(dict(permission=False,diagnostic_only=diagnostic_orientations is not None,position=p.tolist(),quaternion_wxyz=q.tolist(),object_center=center.tolist(),
                attachment_local=local.tolist(),rotation=rot.tolist(),half_extent=half.tolist(),**quality))
        groups.append(group)
    # Interleave yaw before trying many near-identical translations.
    return [dict(candidate_id='candidate-'+str(i),**row) for i,row in enumerate(r for batch in zip(*groups) for r in batch)] if groups else []


def search_placements(skills,row,shape,reg,destination,*,max_positions=12):
    from .region_geometry import region_cells
    region=region_cells(destination,[2,2])[int(row['target'])]
    candidates=placement_candidates(skills.held,shape,reg,region,skills.profile,row.get('preferred_xy'),row.get('quaternion_wxyz'),max_positions=max_positions)
    tested=[];selected=None
    for candidate in candidates:
        p=np.asarray(candidate['position']);q=np.asarray(candidate['quaternion_wxyz']);checks=[]
        for phase,target in [('release_pose',p),('above_target',p+[0,0,.2])]:
            check=skills.reachable.query(target,q) if hasattr(skills.reachable,'query') else dict(status='KINEMATIC_SOLUTION' if skills.reachable(target,q) else 'NO_SOLUTION_FOUND')
            checks.append(dict(phase=phase,**{k:v for k,v in check.items() if k!='iterations'}))
            if check['status']!='KINEMATIC_SOLUTION':break
        passed=len(checks)==2 and all(x['status']=='KINEMATIC_SOLUTION' for x in checks)
        tested.append(dict(candidate,checks=checks,status='KINEMATIC_CANDIDATE' if passed else 'NO_SOLUTION_FOUND'))
        if passed:selected=candidate;break
    return dict(status='KINEMATIC_CANDIDATE' if selected else 'NO_SOLUTION_FOUND',selected=selected,
        tested=tested,candidate_count=len(candidates),remaining_candidates=len(candidates)-len(tested),
        region=region,source=skills.scene.source,permission=False,physical_unreachable=False,
        path_status='NOT_VERIFIED',limitations=['endpoint IK is not a collision-free path','current hold and final release must be reverified'])


def choose_joint_grasp(skills,name,shape,position,candidates):
    """Bounded search over qualified grasps; hypothetical placement is not a permit."""
    from types import SimpleNamespace
    from .vision_skills import pose_matrix
    target=getattr(getattr(skills,'preplanner',None),'targets',{}).get(name)
    if target is None:return (candidates[0] if candidates else None),[]
    destination=skills.region('destination');results=[]
    for candidate in candidates[:3]:
        q=candidate['q']
        h=dict(object=name,shape=shape,tcp=pose_matrix(np.r_[position,q]),grasp_quaternion=q)
        future=search_placements(SimpleNamespace(held=h,scene=skills.scene,profile=skills.profile,reachable=skills.reachable),
            dict(target=target),shape,dict(residual_95_m=0.),destination,
            max_positions=getattr(skills,'future_search_positions',2))
        results.append(dict(candidate_id=candidate['id'],future_placement=future,
            assumption='hypothetical rigid attachment; must be reverified after actual grasp',permission=False))
        if future['selected']:return candidate,results
    return None,results


class SkillPreplanner:
    def __init__(self,skills):self.s=skills;self.cache={};self.records={}

    def context(self):
        s=self.s
        return digest(plain([getattr(s.scene,'observation_key',s.scene.source),s.motion_version,s.phase,s.effects,
            s.scene.obs.get('robot_joint_pos'),s.scene.obs.get('robot_cartesian_pos'),s.scene.obs.get('robot_gripper_joint_pos'),
            None if s.held is None else [s.held['object'],s.held['tcp'],s.held['shape']],s.profile]))

    def preview(self,row):
        began=time.monotonic()
        count=row.get('future_positions',2)
        if type(count) is not int or not 1<=count<=12:raise ValueError('future_positions must be an integer from 1 to 12')
        s=self.s;key=digest([self.context(),plain(row)])
        if key in self.cache:return dict(copy.deepcopy(self.cache[key]),cache_hit=True,
            original_compute_seconds=self.cache[key]['seconds'],seconds=time.monotonic()-began,new_solver_calls=0)
        started=time.monotonic();result=dict(status='CAPABILITY_CONTRACT_ONLY',permission=False,physical_side_effects=False,
            skill=row['skill'],object=row.get('object'),target=row.get('target'),context=self.context(),cache_hit=False)
        from .skill_contracts import SkillBoundaryStop
        previous=s.__dict__.get('active_condition')
        previous_count=getattr(s,'future_search_positions',2);s.future_search_positions=count
        try:
            if row['skill']=='acquire':
                g,p,q,roles=s._acquire_plan(row['object']);s.plans[row['object']]=(s.scene.serial,(g,p,q,roles))
                result.update(status='KINEMATIC_CANDIDATE',selected=dict(position=plain(p),quaternion_wxyz=plain(q),candidate_id=getattr(s,'selected_grasp_candidate',{}).get(row['object'])),
                    path_status='APPROACH_VISIBLE_POINTS_ONLY',joint_feasibility='future attachment uncertain; must revalidate after grasp')
                result['raw_grasp_candidates']=plain(getattr(s,'raw_grasp_candidates',{}).get(row['object'],[]))
                target=getattr(self,'targets',{}).get(row['object'])
                if target is not None:
                    joint=getattr(s,'joint_grasp_search',{}).get(row['object'],[])
                    selected=getattr(s,'selected_grasp_candidate',{}).get(row['object'])
                    chosen=next((x for x in joint if x['candidate_id']==selected),None)
                    if chosen:
                        result['future_placement']=dict(chosen['future_placement'],assumption=chosen['assumption'],permission=False)
                        result['joint_feasibility']='CONDITIONAL'
                    else:
                        result['joint_feasibility']='NOT_TESTED_no_registered_joint_search'

            elif row['skill']=='deposit':
                if s.held is None or s.held['object']!=row['object']:raise ValueError('Required verified attachment missing')
                shape,reg=s.current_held();destination=s.region('destination')
                result.update(search_placements(s,row,shape,reg,destination))
        except (ValueError,SkillBoundaryStop) as exc:
            if isinstance(exc,SkillBoundaryStop) and hasattr(s,'validation_sink'):s.validation_sink(exc.effects,str(exc))
            result.update(status='EVIDENCE_OR_CAPABILITY_MISSING',reason=str(exc))
        finally:s.active_condition=previous;s.future_search_positions=previous_count
        if row['skill']=='acquire':
            result['joint_search']=copy.deepcopy(getattr(s,'joint_grasp_search',{}).get(row['object'],[]))
            result['future_search']=dict(max_positions=count,max_grasps=3,yaws_per_position=4,
                can_expand=count<12,physical_unreachable=False,
                next_action='Query joint_search; optionally increase future_positions within 1..12' if count<12 else 'Bounded library exhausted; different capability or evidence needed')
            result['physical_unreachable']=False
        result.update(id='search-'+key,seconds=time.monotonic()-started)
        self.records[result['id']]=copy.deepcopy(result);self.cache[key]=copy.deepcopy(result)
        return result


def validate_trajectory(trajectory,start,target,kinematics,*,collision_evidence=None,held=False):
    """Common result validation, never trusts an optimizer's success flag."""
    start=np.asarray(start,float);path=np.asarray(trajectory,float);target=np.asarray(target,float)
    result=dict(status='REJECTED',permission=False,scope='planned_path_requires_live_dispatch_recheck')
    if path.ndim!=2 or len(path)<2 or path.shape[1:]!=start.shape or not np.isfinite(path).all():
        return dict(result,reason='invalid_joint_path')
    try:
        for q in path:kinematics.validate(q)
        end=kinematics.pose(path[-1])
    except ValueError as exc:return dict(result,reason=str(exc))
    pos=float(np.linalg.norm(end[4:]-target[4:]));angle=float(np.degrees(2*np.arccos(np.clip(abs(np.dot(end[:4],target[:4])),0,1))))
    error=float(np.max(abs(path[0]-start)));result.update(start_joint_error_rad=error,position_error_m=pos,orientation_error_deg=angle)
    if error>.002:return dict(result,reason='trajectory_does_not_start_at_current_joints')
    if pos>.001 or angle>1:return dict(result,reason='trajectory_endpoint_failed_original_IK_tolerances')
    if not collision_evidence or collision_evidence.get('status')!='PASS' or not collision_evidence.get('complete_public_world'):
        return dict(result,status='UNKNOWN',reason='missing_public_collision_or_free_space_evidence')
    if held and not collision_evidence.get('attachment_included'):
        return dict(result,status='UNKNOWN',reason='held_volume_missing_from_collision_model')
    return dict(result,status='VALIDATED_CANDIDATE',reason='start_endpoint_and_supplied_collision_evidence_checked')


def planning_summary(result):
    """Default tool response fits an agent decision; details remain indexed."""
    keys=('id','status','skill','object','target','context','permission','physical_side_effects',
          'selected','candidate_count','remaining_candidates','path_status','reason','cache_hit',
          'seconds','original_compute_seconds','new_solver_calls','joint_feasibility','limitations','future_search','physical_unreachable')
    out={k:copy.deepcopy(result[k]) for k in keys if k in result}
    out['permission']=False
    out['detail_tool']='planning_query(id); do not pass this search ID to run_plan or inputs'
    if 'future_placement' in result:
        f=result['future_placement'];out['future_placement']={k:copy.deepcopy(f.get(k)) for k in ('status','selected','candidate_count','remaining_candidates','assumption')}
    tested=result.get('tested',[])
    if tested:
        out['last_failure_checks']=[{k:c.get(k) for k in ('id','phase','status','position_error_m','orientation_error_deg','failure_type')} for c in tested[-1].get('checks',[])]
    return out
