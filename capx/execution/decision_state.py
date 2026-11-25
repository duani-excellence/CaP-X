"""Bounded decision projection over existing public state, not raw Belief dump."""
import numpy as np
from .agent_events import plain


def encode_decision_state(state,skills,belief,searches,budget,failures):
    # A checkpoint precedes the first capture. Missing measurement remains
    # unknown; encoding must not capture or fabricate evidence as a side effect.
    observation=getattr(skills.scene,'obs',{})
    remaining=state.get('remaining_goals',[]);objects={}
    all_goals=state.get('task_goals',remaining)
    names={k.split(':',1)[1] for k in remaining if ':' in k}
    if skills.held:names.add(skills.held['object'])
    for name in sorted(names):
        b=belief.get('object:'+name,{}) or {};v=b.get('values',{})
        objects[name]=dict(geometry={k:v[k] for k in ('center','height_m','half_extent_xy','bottom_z','top_z','uncertainty') if k in v},
            evidence_kind=b.get('kind','UNKNOWN'),source=b.get('source'),id=b.get('id'),permission=False)
    attachment=None
    if skills.held:
        h=skills.held;t=np.asarray(h['tcp']);c=np.asarray(h['shape']['center']);local=np.linalg.inv(t)@np.r_[c,1]
        nominal=np.eye(4);nominal[:3,3]=c
        basis=np.array(h['shape'].get('basis_xy',np.eye(2)))
        if basis.shape==(2,2) and np.allclose(basis.T@basis,np.eye(2),atol=1e-5):
            nominal[:2,:2]=basis
            if np.linalg.det(nominal[:3,:3])<0:nominal[:3,1]*=-1
        nominal_tcp_object=np.linalg.inv(t)@nominal
        attachment=dict(T_tcp_shape_frame=nominal_tcp_object.tolist(),frame_semantics='nominal principal-axis shape frame at registered grasp, not a fresh pose measurement',object=h['object'],object_center_in_tcp=local[:3].tolist(),reference_tcp=t.tolist(),
            current_status=state.get('conditions',{}).get('held:'+h['object'],{}).get('truth','UNKNOWN'),
            source=belief.get('attachment:'+h['object'],{}).get('source'),requires='fresh public attachment registration before action')
    relations=[]
    for search in searches[-4:]:
        name=search.get('object');region=search.get('region',search.get('future_placement',{}).get('region'))
        b=belief.get('object:'+str(name),{});shape=b.get('values',{});kind=b.get('kind','UNKNOWN')
        attached=belief.get('attachment:'+str(name),{})
        if skills.held and skills.held['object']==name and attached.get('values',{}).get('shape'):
            shape=attached['values']['shape'];b=attached;kind=attached.get('kind','UNKNOWN')
        if region and all(k in region for k in ('lo','hi')) and 'center' in shape:
            center=(np.array(region['lo'])+np.array(region['hi']))/2
            relations.append(dict(object=name,target=search.get('target'),
                delta_xy_to_target_center=(center-np.array(shape['center'])[:2]).tolist(),
                target_bounds={k:region[k] for k in ('lo','hi','floor_z') if k in region},
                object_evidence_kind=kind,object_source=b.get('source'),target_source=search.get('source'),
                permission=False,meaning='relation between indexed estimates, not a current collision or release permit'))
    summaries=[]
    for r in searches[-4:]:
        tested=r.get('tested',[])
        summaries.append({**{k:r.get(k) for k in ('id','status','skill','object','target','selected','candidate_count','remaining_candidates','path_status','reason','cache_hit')},
            'decision_status':{'NO_SOLUTION_FOUND':'NO_SOLUTION_IN_TESTED_SET','KINEMATIC_CANDIDATE':'IK_FEASIBLE',
                'EVIDENCE_OR_CAPABILITY_MISSING':'INSUFFICIENT_GEOMETRIC_EVIDENCE'}.get(r.get('status'),r.get('status','UNKNOWN')),
            'release_status':'RELEASE_UNVERIFIED',
            'tested_pose_range':dict(positions=[t.get('position') for t in tested],
                quaternion_wxyz=list({tuple(t['quaternion_wxyz']) for t in tested if 'quaternion_wxyz' in t})),
            'untested_scope':'Other grasps/attachments, paths and capability-supported orientations are not disproven by this bounded search',
            'tested_summary':[dict(candidate_id=t['candidate_id'],position=t['position'],quaternion_wxyz=t['quaternion_wxyz'],status=t['status'],
                checks=[{k:c.get(k) for k in ('id','phase','status','position_error_m','orientation_error_deg','failure_type','cache_hit')} for c in t['checks']]) for t in tested[-3:]],
            'detail_tool':'planning_query(search_id) / planning_query(ik_id)'})
    # The range is bounded by the actual search (48 candidates); detailed
    # iterations remain queryable and are never copied into the default prompt.
    return plain(dict(schema='capx.decision_state.v36',permission=False,
        coordinates=dict(frame='robot_base',position_units='m',angle_units='radians unless explicitly deg',quaternion_order='WXYZ',pose_reference='TCP, not hand origin'),
        goals=dict(remaining=remaining,verified=[k for k,v in state.get('conditions',{}).items() if v.get('truth')=='TRUE' and k in all_goals and k not in remaining]),
        phase=skills.phase,objects=objects,attachment=skills.held['object'] if skills.held else None,
        attachment_estimate=attachment,spatial_relations=relations,
        interaction=dict(gripper=observation.get('robot_gripper_joint_pos'),
            force_observation=observation.get('wrist_ft'),
            occupied_resources=state.get('locks',[]),interpretation='Stable force is not proof of precise attachment pose or absence of slip'),
        placement_quality={k.split(':',1)[1]:{f:v.get(f) for f in ('values','source','kind','status')} for k,v in belief.items() if k.startswith('quality:')},
        planning_statistics=dict(unique_searches=len({r.get('id') for r in searches}),
            cache_hits=getattr(getattr(skills,'reachable',None),'hits',None),
            unique_ik_targets=len(getattr(getattr(skills,'reachable',None),'records',{})),
            interpretation='API requests, candidate checks and actual solver iterations are distinct'),
        available_bindings=state.get('available_bindings',[]),
        region_evidence={k:{field:v.get(field) for field in ('id','kind','status','source')} for k,v in belief.items() if k.startswith('region:')},
        robot=dict(tcp=observation.get('robot_cartesian_pos'),joints=observation.get('robot_joint_pos'),source=getattr(skills.scene,'source',None)),
        unmet_conditions={k:v.get('truth') for k,v in state.get('conditions',{}).items() if v.get('truth')!='TRUE'},
        plans=summaries,failures=failures[-3:],remaining_budget=budget,
        options=['preview_skill: compare candidates without motion','planning_query: inspect failed residuals and untested space','run_plan: request live-validated dispatch','observe only missing evidence'],
        interpretation='NO_SOLUTION_FOUND is bounded solver failure, not physical unreachability; endpoint IK never authorizes motion'))
