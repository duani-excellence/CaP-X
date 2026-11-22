"""Independent offline truth labels, recovery effects and deduplicated billing.

Never imported by an online monitor. Release labels are geometric risk proxies,
not proof that a counterfactual release would succeed. Original physical reward
is reported separately and never recomputed or relaxed here.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from capx.experiments.feedback_analysis import truth_tracking,truth_release,extent
from capx.experiments.graph_statistics import success as policy_success,physical_success as original_physical_success,common_eligible_success

CHECK_TYPES={'tcp_tracking':'control','tcp_tracking_retry':'control','local_alignment_control':'control',
             'pre_release':'release','hold_probe':'hold','lift':'hold','table_clearance':'table',
             'precise_tcp':'control','precise_tcp_pre_descent':'control','precise_tcp_command':'command'}


def read(path):return json.loads(Path(path).read_text())


def color(name):
    if name in ('red','red cube'):return 'red'
    if name in ('green','green cube'):return 'green'
    return None


def legacy_tracking(state,geometry):
    canonical=state['canonical'];base=geometry['base_world']
    return {'tcp':canonical['tcp_world']},{'base_position_world':base['position'],'base_rotation_world':base['rotation']}


def release_risk(state,geometry,held,placement):
    canonical=state['canonical'];objects=canonical['objects'];obj=objects[held]
    if placement=='table_place':
        p=np.asarray(obj['position_world']);half=extent(obj['quaternion_wxyz'],geometry['objects'][held]['half_size'])
        z=geometry['table_top_world_z'];center=np.asarray(geometry['table_center_world']);table_half=np.asarray(geometry['table_half_extent_xy'])
        if z is None or center.size<2 or table_half.size!=2:raise ValueError('Unknown physical table boundary')
        gap=float(p[2]-half[2]-z);inside=bool(np.all(np.abs(p[:2]-center[:2])+half[:2]<=table_half))
        overlaps=[]
        for other,value in objects.items():
            if other==held:continue
            other_half=extent(value['quaternion_wxyz'],geometry['objects'][other]['half_size'])
            if np.all(np.abs(p-np.asarray(value['position_world']))<half+other_half):overlaps.append(other)
        held_contact=bool(obj['bilateral_contact'])
        return dict(xy_error_m=0. if inside else None,bottom_gap_m=gap,held=held_contact,inside_table=inside,
                    overlapping_objects=overlaps,truth_bad=not inside or bool(overlaps) or gap>.020 or gap<-.006 or not held_contact,
                    label_source='actual_table_and_object_AABBs_plus_bilateral_contact')
    if placement!='place':raise ValueError('Placement skill unknown')
    support='green' if held=='red' else 'red';other=objects[support]
    converted={'objects':{'primary':{'position':obj['position_world'],'quaternion_wxyz':obj['quaternion_wxyz']},
                         'secondary':{'position':other['position_world'],'quaternion_wxyz':other['quaternion_wxyz']}},
               'grasp_contact':bool(obj['bilateral_contact'])}
    sizes={'red_half_size':geometry['objects'][held]['half_size'],'green_half_size':geometry['objects'][support]['half_size']}
    return dict(truth_release(converted,sizes),label_source='physical_object_AABBs_plus_bilateral_contact')


def table_footprint_risk(state,geometry,held,region):
    """Independent task-object/table proxy, NOT a full swept-volume proof.

    The graph supplies only which public footprint was checked. Its online mask,
    PASS flag and eventual reward never provide the independent truth label.
    Retained historical node data remain usable even after later invalidation.
    """
    if not isinstance(region,dict) or region.get('type')!='table_region' or region.get('frame')!='robot_base':
        raise ValueError('Missing recorded robot-base table region')
    center=np.asarray(region.get('center'),float);half=np.asarray(region.get('footprint_half_extent_xy'),float)
    base=geometry['base_world'];rotation=np.asarray(base['rotation'],float);translation=np.asarray(base['position'],float)
    table_center=np.asarray(geometry['table_center_world'],float);table_half=np.asarray(geometry['table_half_extent_xy'],float)
    table_z=geometry['table_top_world_z']
    if (center.shape!=(3,) or half.shape!=(2,) or rotation.shape!=(3,3) or translation.shape!=(3,)
        or table_center.shape!=(3,) or table_half.shape!=(2,) or table_z is None
        or not np.isfinite(np.r_[center,half,rotation.ravel(),translation,table_center,table_half,table_z]).all()
        or np.any(half<=0) or np.any(table_half<=0)
        or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(rotation),1)):
        raise ValueError('Invalid independent table/footprint calibration')
    corners=np.array([[center[0]+x*half[0],center[1]+y*half[1],center[2]] for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))])
    world=corners@rotation.T+translation
    if np.ptp(world[:,2])>1e-8:raise ValueError('Table footprint is not parallel to independently known table')
    inside=bool(np.all(np.abs(world[:,:2]-table_center[:2])<=table_half))
    objects=state['canonical']['objects']
    if held not in objects:raise ValueError('Unknown independently recorded held object identity')
    overlap=[]
    # Separating axes for the public rectangle and each private world AABB.
    # AABBs conservatively bound rotated physical cubes; this is a risk proxy.
    axes=[np.array([1.,0.]),np.array([0.,1.])]
    for edge in (world[1,:2]-world[0,:2],world[3,:2]-world[0,:2]):
        axes.append(np.array([-edge[1],edge[0]])/np.linalg.norm(edge))
    for name,obj in objects.items():
        if name==held:continue
        position=np.asarray(obj['position_world'],float)
        bounds=extent(obj['quaternion_wxyz'],geometry['objects'][name]['half_size'])
        if position.shape!=(3,) or not np.isfinite(np.r_[position,bounds]).all():raise ValueError('Invalid independent object geometry')
        if position[2]+bounds[2]<=table_z+.004:continue
        box=np.array([[position[0]+x*bounds[0],position[1]+y*bounds[1]] for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))])
        separated=any(np.max(world[:,:2]@axis)<np.min(box@axis) or np.max(box@axis)<np.min(world[:,:2]@axis) for axis in axes)
        if not separated:overlap.append(name)
    return dict(truth_bad=not inside or bool(overlap),inside_table=inside,overlapping_objects=overlap,
        held_object_excluded=held,footprint_world=world.tolist(),estimated_plane_error_m=float(world[0,2]-table_z),
        label_source='private_table_boundary_and_current_nonheld_task_object_AABBs',
        label_scope='table_footprint_geometric_risk_proxy',
        limitation='Not whole-arm/path collision certification, estimated table height certification or counterfactual success. Held object and known robot are excluded; AABBs conservatively bound rotated task objects across the vertical column, including elevated nonheld objects. An epistemic block can be a false_block against this physical proxy.')


def label_check(event,private,geometry,*,baseline_z=None,graph_nodes=None):
    category=CHECK_TYPES.get(event.get('check'))
    item=dict(check=event.get('check'),category=category,status=event.get('status'),reason=event.get('reason'),
              uncertain=event.get('status')=='UNCERTAIN',classification='unknown')
    metrics=event.get('metrics',{})
    item.update(stage=metrics.get('stage'),correction=metrics.get('correction'),target_tcp=metrics.get('target_tcp'))
    if category=='command':
        item.update(classification='not_applicable',decision='announcement',command_tcp=metrics.get('command_tcp'),
                    note='Before-motion command announcement, not a blocking detector or physical evidence')
        return item
    if category is None:item['unknown_reason']='not_a_labeled_decision';return item
    if not private or not private.get('state') or geometry is None:item['unknown_reason']='missing_private_state_or_geometry';return item
    try:
        state=private['state'];canonical=state['canonical'];held=color(private.get('held_object'))
        item.update(sim_time=canonical.get('sim_time'),held_object=private.get('held_object'),phase=private.get('phase'),
                    placement_skill=private.get('placement_skill'),local_attempts=private.get('local_attempts'))
        if category=='control':
            converted,base=legacy_tracking(state,geometry);truth=truth_tracking(converted,base,metrics)
            if event['check'].startswith('precise_tcp'):
                # The grounded caller uses these helper defaults; independent FK
                # is compared to the target, never the public actual_tcp claim.
                truth.update(truth_bad=truth['position_error_m']>.003 or truth['orientation_error_rad']>np.deg2rad(5),
                    position_tolerance_m=.003,orientation_tolerance_deg=5.,
                    label_source='independent_simulator_TCP_FK_at_grounded_control_default_tolerances')
        elif category=='release':truth=release_risk(state,geometry,held,private.get('placement_skill'))
        elif category=='table':
            node=(graph_nodes or {}).get(metrics.get('region_id'),{})
            if node.get('kind')!='estimate':raise ValueError('Missing retained table-region graph node')
            truth=table_footprint_risk(state,geometry,held,node.get('data'))
        else:
            obj=canonical['objects'][held];contact=bool(obj['bilateral_contact'])
            truth=dict(held=contact,truth_bad=not contact,label_source='checkpoint_bilateral_fingerpad_contact')
            if event['check']=='lift':
                if held not in (baseline_z or {}):raise ValueError('Missing independent grasp baseline')
                rise=float(obj['position_world'][2]-baseline_z[held])
                truth.update(object_rise_m=rise,truth_bad=not contact or rise<=.035,label_source='bilateral_contact_and_actual_rise_from_grasp_checkpoint')
        truth['truth_bad']=bool(truth['truth_bad'])
        item.update(truth)
        # UNCERTAIN release/lift/hold-probe actually blocks progress. Count the
        # decision separately from detector certainty, never as a confident TP.
        if event['status'] not in ('PASS','FAIL','UNCERTAIN'):raise ValueError('Unknown online status')
        allowed=event['status']=='PASS';item['decision']='allow' if allowed else 'block'
        item['classification']=('false_allow' if truth['truth_bad'] else 'true_allow') if allowed else ('true_block' if truth['truth_bad'] else 'false_block')
    except (KeyError,TypeError,ValueError,IndexError) as exc:
        item['unknown_reason']=str(exc);item['classification']='unknown'
    return item


def risk_score(check):
    if check.get('category')=='release' and check.get('bottom_gap_m') is not None:
        xy=check.get('xy_error_m');gap=check['bottom_gap_m']
        return max(0.,(xy or 0)-.010)+max(0.,gap-.020)+max(0.,-.006-gap)+(.1 if check.get('held') is False else 0.)+(.1 if check.get('inside_table') is False or check.get('overlapping_objects') else 0.)
    if check.get('category')=='control' and check.get('position_error_m') is not None:
        return check['position_error_m']+check['orientation_error_rad']*.05
    return None


def transition_cleared(before, after):
    """None means the independent before/after evidence is unavailable."""
    if not isinstance(before,bool) or not isinstance(after,bool):return None
    return before and not after


def active_evidence_attempts(events, trace_rows, private, existing):
    """Reconstruct admitted active-sensing attempts, including interrupted ones.

    A visibility-clearance check alone is not a dispatched probe. The trusted
    phase gate marks the attempt; only a subsequent same-attempt observation can
    establish public evidence recovery. It cannot establish physical risk repair.
    """
    result=[]
    nested_visibility=set()
    for index,(start,event) in enumerate(events):
        metrics=event.get('metrics') or {}
        if event.get('check')!='semantic_refinement' or not metrics.get('attempted'):continue
        before=metrics.get('initial_validation') or {};after=metrics.get('final_validation') or {}
        result.append(dict(kind='semantic_refinement',role='evidence_validation',
            trigger='failed_or_uncertain_checkpoint',from_event=None,attempt_event=index,to_event=index,
            motion_calls=0,tool_invocations=1,detector_passed=after.get('status')=='PASS',
            detector_cleared=before.get('status') in ('FAIL','UNCERTAIN') and after.get('status')=='PASS',
            physical_risk_cleared=None,physical_error_improved=None,before_risk=None,after_risk=None,
            independent_physical_repair_evidence=False,before=before,after=after))
    # Table reobservation is one public-evidence operation. Its optional wrist
    # probe is a child, and attachment PASS does not certify the table check.
    for index,(start,event) in enumerate(events):
        if event.get('check')!='phase_precondition' or event.get('status')!='PASS' or (event.get('metrics') or {}).get('operation')!='reobserve_table':continue
        held=(private.get(index) or {}).get('held_object') or (event.get('metrics') or {}).get('held_object')
        end_index=next((i for i in range(index+1,len(events))
            if events[i][1].get('check')=='phase_precondition'
            and (events[i][1].get('metrics') or {}).get('operation') not in (None,'visibility_probe')),len(events))
        def matches(i):
            other=(private.get(i) or {}).get('held_object')
            return not held or not other or held==other
        after_index=next((i for i in range(index+1,end_index) if events[i][1].get('check')=='table_clearance' and matches(i)),None)
        prior_index=next((i for i in range(index-1,-1,-1) if events[i][1].get('check')=='table_clearance' and matches(i)),None)
        before=events[prior_index][1] if prior_index is not None else None
        after=events[after_index][1] if after_index is not None else None
        limit=after_index if after_index is not None else end_index
        children=[i for i in range(index+1,limit) if events[i][1].get('check')=='phase_precondition'
            and events[i][1].get('status')=='PASS' and (events[i][1].get('metrics') or {}).get('operation')=='visibility_probe']
        nested_visibility.update(children)
        end=events[limit][0] if limit<len(events) else len(trace_rows)
        trigger=('failed_or_uncertain_checkpoint' if before.get('status') in ('FAIL','UNCERTAIN') else 'normal_verification') if before else 'unknown'
        result.append(dict(kind='table_reobservation',role='evidence_validation',trigger=trigger,
            from_event=prior_index,attempt_event=index,to_event=after_index,held_object=held,
            included_visibility_probes=len(children),
            motion_calls=sum(r.get('kind')=='api' and r.get('name')=='goto_pose' for r in trace_rows[start:end]),
            detector_passed=after.get('status')=='PASS' if after else None,
            detector_cleared=bool(before.get('status') in ('FAIL','UNCERTAIN') and after.get('status')=='PASS') if before and after else None,
            physical_risk_cleared=None,physical_error_improved=None,before_risk=None,after_risk=None,
            independent_physical_repair_evidence=False,before=before,after=after))
    for index,(start,event) in enumerate(events):
        if index in nested_visibility:continue
        operation=(event.get('metrics') or {}).get('operation')
        if event.get('check')!='phase_precondition' or event.get('status')!='PASS' or operation not in ('visibility_probe','verify_hold'):continue
        name='attachment_consistency' if operation=='visibility_probe' else 'hold_probe'
        held=(private.get(index) or {}).get('held_object') or (event.get('metrics') or {}).get('held_object')
        def same_object(i):
            other=(private.get(i) or {}).get('held_object')
            return not held or not other or held==other
        end_index=next((i for i in range(index+1,len(events)) if events[i][1].get('check')=='phase_precondition' and (events[i][1].get('metrics') or {}).get('operation') is not None),len(events))
        after_index=next((i for i in range(index+1,end_index) if events[i][1].get('check')==name and same_object(i)),None)
        if operation=='verify_hold' and after_index is not None and any(r.get('to_event')==after_index and r['kind']=='hold_evidence_probe' for r in existing):continue
        prior_names={'attachment_consistency'} if operation=='visibility_probe' else {'lift','hold_probe'}
        prior_index=next((i for i in range(index-1,-1,-1) if events[i][1].get('check') in prior_names and same_object(i)),None)
        before=events[prior_index][1] if prior_index is not None else None
        after=events[after_index][1] if after_index is not None else None
        end=events[after_index][0] if after_index is not None else events[end_index][0] if end_index<len(events) else len(trace_rows)
        trigger=('failed_or_uncertain_checkpoint' if before.get('status') in ('FAIL','UNCERTAIN') else 'normal_verification') if before else 'unknown'
        result.append(dict(kind='active_visibility' if operation=='visibility_probe' else 'hold_evidence_probe',
            role='evidence_validation',trigger=trigger,from_event=prior_index,attempt_event=index,to_event=after_index,
            held_object=held,motion_calls=sum(r.get('kind')=='api' and r.get('name')=='goto_pose' for r in trace_rows[start:end]),
            detector_passed=after.get('status')=='PASS' if after else None,
            detector_cleared=bool(before.get('status') in ('FAIL','UNCERTAIN') and after.get('status')=='PASS') if before and after else None,
            physical_risk_cleared=None,physical_error_improved=None,before_risk=None,after_risk=None,
            physical_holding_verified=None,independent_physical_repair_evidence=False,
            before=before,after=after))
    return result


def intervention_records(recovery,result):
    """Orthogonal roles; an evidence probe is never itself a physical repair."""
    physical_success=(result.get('info') or {}).get('physical_success',result.get('reward')==1)
    common=dict(final_task_success=bool(policy_success(result)),final_physical_task_success=bool(physical_success))
    records=[]
    for item in recovery:
        before=item.get('before') or {}
        trigger=item.get('trigger') or ('failed_or_uncertain_checkpoint' if before.get('status') in ('FAIL','UNCERTAIN') else 'normal_verification' if before.get('status')=='PASS' else 'unknown')
        records.append(dict(item,role='evidence_acquisition' if item['role']=='evidence_validation' else 'physical_correction',
                            trigger=trigger,**common))
    decisions={d['block']:d for d in result.get('recovery_decisions',[]) if 'block' in d}
    blocks={b['block']:b for b in result.get('blocks',[]) if 'block' in b}
    for number in range(1,int(result.get('replans',0))+1):
        decision=decisions.get(number,{});block=blocks.get(number)
        records.append(dict(kind='llm_replan',role='llm_replan',trigger='unfinished_episode',block=number,
            decision=decision.get('decision'),request=decision.get('request'),code_executed=block is not None,
            block_physical_success=(block.get('reward')==1) if block else None,
            block_policy_success=block.get('policy_success') if block else None,
            detector_passed=None,detector_cleared=None,physical_risk_cleared=None,physical_error_improved=None,**common))
    return records


def analyze_episode(path):
    path=Path(path);result=read(path);trace=Path(result.get('trace_path',''))
    trace_rows=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()] if trace.is_file() else []
    geometry=read(path.parent/'evaluation-geometry.json') if (path.parent/'evaluation-geometry.json').exists() else None
    events=[(i,row) for i,row in enumerate(trace_rows) if row.get('kind')=='feedback']
    private={row['event_index']:row for row in trace_rows if row.get('kind')=='feedback_evaluation'}
    graph_file=path.parent/'graph-final.json'
    graph_nodes=read(graph_file).get('nodes',{}) if graph_file.is_file() else {}
    baseline={};checks=[]
    for index,(trace_index,event) in enumerate(events):
        p=private.get(index)
        if event.get('check')=='grasp' and p:
            held=color(p.get('held_object'))
            try:baseline[held]=p['state']['canonical']['objects'][held]['position_world'][2]
            except (KeyError,TypeError):pass
        if event.get('check') not in CHECK_TYPES:continue
        check=label_check(event,p,geometry,baseline_z=baseline,graph_nodes=graph_nodes)
        check.update(event_index=index,trace_index=trace_index);checks.append(check)
    recovery=[]
    for j,current in enumerate(checks):
        if current['check']=='pre_release':
            prior=next((c for c in reversed(checks[:j]) if c['check']=='pre_release' and c.get('held_object')==current.get('held_object')),None)
            if not prior or current.get('local_attempts') is None or prior.get('local_attempts') is None or current['local_attempts']<=prior['local_attempts']:continue
            kind='release_alignment'
        elif current['check']=='tcp_tracking_retry':
            prior=next((c for c in reversed(checks[:j]) if c['check']=='tcp_tracking'),None);kind='control_retry'
        elif current['check']=='precise_tcp' and current.get('correction') is True:
            if current.get('reason') in ('correction_limit','offset_limit','correction_budget','motion_callback_error','broker_interrupted'):continue
            prior=next((c for c in reversed(checks[:j]) if c['check']=='precise_tcp' and c.get('stage')==current.get('stage') and c.get('target_tcp')==current.get('target_tcp') and c.get('held_object')==current.get('held_object')),None)
            kind='precision_control'
        elif current['check']=='hold_probe':
            prior=next((c for c in reversed(checks[:j]) if c['category']=='hold' and c.get('held_object')==current.get('held_object')),None);kind='hold_evidence_probe'
        else:continue
        if prior is None:continue
        before,after=risk_score(prior),risk_score(current)
        motions=sum(row.get('kind')=='api' and row.get('name')=='goto_pose' for row in trace_rows[prior['trace_index']:current['trace_index']])
        recovery.append(dict(kind=kind,from_event=prior['event_index'],to_event=current['event_index'],motion_calls=motions,
            detector_passed=current['status']=='PASS',
            detector_cleared=bool(prior['status'] in ('FAIL','UNCERTAIN') and current['status']=='PASS'),
            physical_risk_cleared=transition_cleared(prior.get('truth_bad'),current.get('truth_bad')),
            physical_error_improved=(after<before-1e-6) if before is not None and after is not None else None,
            before_risk=before,after_risk=after,physical_holding_verified=current.get('held') if kind=='hold_evidence_probe' else None,
            role='evidence_validation' if kind=='hold_evidence_probe' else 'physical_correction',
            before=prior,after=current))
    recovery.extend(active_evidence_attempts(events,trace_rows,private,recovery))
    interventions=intervention_records(recovery,result)
    excluded=Counter(e.get('check','unknown') for _,e in events if e.get('check') not in CHECK_TYPES)
    coverage=dict(all_feedback_events=len(events),eligible_checkpoints=sum(c['category']!='command' for c in checks),
        independently_labeled_checkpoints=sum(c['classification'] not in ('unknown','not_applicable') for c in checks),
        unknown_checkpoints=sum(c['classification']=='unknown' for c in checks),
        command_announcements=sum(c['category']=='command' for c in checks),excluded_by_check=dict(excluded))
    counts=Counter(c['classification'] for c in checks if c['classification']!='not_applicable')
    by_type={category:dict(Counter(c['classification'] for c in checks if c['category']==category)) for category in ('release','control','hold','table')}
    ineffective=sum(r['role']=='physical_correction' and r['physical_error_improved'] is False and not r['physical_risk_cleared'] for r in recovery)
    return dict(task=result['task'],seed=result['seed'],repeat=result.get('repeat',0),group=result['group'],fault=result.get('fault','none'),ablation=result.get('ablation','full'),
        result_path=str(path.resolve()),source=result.get('source','model_runner'),online_vision=result.get('online_vision'),llm_requests=result.get('llm_requests'),task_success=bool(policy_success(result)),policy_success=bool(policy_success(result)),
        policy_outcome_available=result.get('status') is not None,physical_reward=result.get('reward'),common_eligible_success=bool(common_eligible_success(result)),raw_physical_is_task_success=False,completion_contract=result.get('completion_contract','legacy_unspecified'),
        physical_task_success=bool(original_physical_success(result)) if original_physical_success(result) is not None else None,
        physical_success_source='info.physical_success' if 'physical_success' in (result.get('info') or {}) else 'original_reward',status=result.get('status'),stop_reason=result.get('stop_reason'),
        interventions=interventions,intervention_trace_available=trace.is_file(),detection_coverage=coverage,
        detection=dict(counts),detection_by_type=by_type,uncertain_checks=sum(c['uncertain'] for c in checks if c['classification']!='not_applicable'),
        command_announcements=sum(c['category']=='command' for c in checks),
        conditional_evidence=[dict(event_index=i,check=e.get('check'),status=e.get('status'),reason=e.get('reason'),metrics=e.get('metrics'),independent_truth_evidence=False) for i,(_,e) in enumerate(events) if e.get('check')=='attachment_consistency' or 'conditional' in str(e.get('check',''))],checks=checks,recoveries=recovery,
        invalid_recovery_actions=ineffective,recovery_evidence_unknown=sum(r['physical_error_improved'] is None for r in recovery if r['role']=='physical_correction'),
        replan_physical_success_blocks=sum(b.get('block',0)>0 and b.get('reward')==1 for b in result.get('blocks',[])),
        request_paths=result.get('request_paths',[]),raw_result=result)


def billing(roots,episodes):
    paths=set();unknown=[]
    for row in episodes:
        for source in row.get('request_paths',[]):
            p=Path(source)/'receipt.json'
            if p.exists():paths.add(p.resolve())
    for root in roots:
        for ledger_path in Path(root).rglob('request-ledger.json'):
            ledger=read(ledger_path)
            for row in ledger.get('requests',[]):
                directory=Path(row.get('directory',''))
                receipts=list(directory.glob('llm/*/receipt.json')) if directory.is_dir() else []
                if receipts:paths.update(p.resolve() for p in receipts)
                else:unknown.append(dict(identity=str(ledger_path.resolve())+':'+row['label'],reserved_usd=row.get('cost_upper_usd',row.get('reserved_usd',0)),state=row.get('state')))
    seen=set();records=[]
    for p in sorted(paths):
        r=read(p);identity='response:'+r['response_id'] if r.get('response_id') else 'request:'+r['request_id'] if r.get('request_id') else str(p)
        if identity in seen:continue
        seen.add(identity);usage=r.get('usage') or {};known=all(isinstance(usage.get(k),(int,float)) for k in ('prompt_tokens','completion_tokens'))
        hit=usage.get('prompt_cache_hit_tokens',0) if known else None;miss=usage.get('prompt_cache_miss_tokens',usage['prompt_tokens']-hit) if known else None
        records.append(dict(identity=identity,receipt=str(p),known_usage=known,input_tokens=usage.get('prompt_tokens') if known else None,
            output_tokens=usage.get('completion_tokens') if known else None,reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',usage.get('reasoning_tokens')),
            cost_upper_usd=r.get('cost_upper_usd',0),historical_peak_estimate_usd=(hit*.044+miss*1.32+usage['completion_tokens']*3.96)/1e6 if known else None,
            seconds=r.get('seconds'),http_status=r.get('http_status')))
    unique_unknown={r['identity']:r for r in unknown}
    return dict(unique_requests=len(records)+len(unique_unknown),confirmed_receipts=len(records),unknown_usage_requests=sum(not r['known_usage'] for r in records)+len(unique_unknown),
        input_tokens=sum(r.get('input_tokens') or 0 for r in records),output_tokens=sum(r.get('output_tokens') or 0 for r in records),
        reasoning_tokens=sum(r.get('reasoning_tokens') or 0 for r in records),reasoning_is_already_in_output=True,
        cost_upper_usd=sum(r.get('cost_upper_usd') or 0 for r in records)+sum(r['reserved_usd'] for r in unique_unknown.values()),
        known_usage_historical_peak_estimate_usd=sum(r.get('historical_peak_estimate_usd') or 0 for r in records),
        known_usage_cache_aware_estimate_range_usd=dict(peak=sum(r.get('historical_peak_estimate_usd') or 0 for r in records),
            offpeak=sum(r.get('historical_peak_estimate_usd') or 0 for r in records)/2),
        request_seconds=sum(r.get('seconds') or 0 for r in records),receipts=records,uncertain_requests=list(unique_unknown.values()),
        pricing_note='DeepSeek pricing verified by the root audit on 2026-11-20: https://api-docs.deepseek.com/quick_start/pricing/ ; V4 Pro peak input hit/miss $0.044/$1.32 per M, output $3.96 per M; offpeak half. Usage-based estimates are not an invoice. cost_upper_usd remains the existing all-input-miss reservation/accounting upper bound. Shared request IDs/receipt paths billed once.')


def analyze(roots,*,n_bootstrap=2000):
    from capx.experiments.graph_statistics import analyze as cluster_analysis
    roots=list(dict.fromkeys(Path(p).resolve() for p in roots));paths=set()
    for root in roots:paths.update(p.resolve() for p in root.glob('**/episodes/*/result.json'))
    episodes=[analyze_episode(p) for p in sorted(paths)];enriched=[]
    for item in episodes:
        row=dict(item['raw_result']);row['detection']=item['detection'];row['invalid_recovery_actions']=item['invalid_recovery_actions'];row['interventions']=item['interventions'];row['detection_coverage']=item['detection_coverage'];enriched.append(row)
    comparisons=cluster_analysis(enriched,n_bootstrap=n_bootstrap)
    for item in episodes:del item['raw_result']
    return dict(schema='capx.graph.offline_evaluation.v1',comparisons=comparisons,episodes=episodes,billing=billing(roots,episodes),
        notes=['All truth is offline; this module must never be imported by online monitors.',
        'Release true/false labels are geometric risk proxies (>10mm XY, >20mm gap, <-6mm overlap or absent bilateral contact), not counterfactual-success guarantees.',
        'Table placement uses actual table boundary, held AABB, other-object overlap and bilateral contact; unsupported table geometry remains unknown.',
        'Hold uses object-specific bilateral contact; Lift also requires >35mm actual rise from last independent grasp checkpoint. Physical task outcome retains the original evaluator flag/reward; policy task_success also requires clean completion within the recorded budget.',
        'precise_tcp and precise_tcp_pre_descent use independent FK with grounded defaults 3mm/5deg. precise_tcp_command is only an announcement, excluded from detector confusion counts.',
        'Table-clearance labels use the recorded public footprint and independent current nonheld task-object AABBs/table boundary. They are geometric risk proxies, not whole-arm/path collision or counterfactual success; unknown geometry remains unknown.',
        'Conditional attachment/support claims are not independent physical truth and never replace missing canonical evaluator state.',
        'UNCERTAIN actual blocking is labeled separately from certainty. It is not automatically counted as correct detection.',
        'Legacy recovery counts include local active sensing and LLM replanning; they are not counts of physical repair. New intervention roles separate physical_correction, evidence_acquisition and llm_replan. Normal initial verify_hold remains normal verification.',
        'Detection confusion is limited to CHECK_TYPES checkpoints with independent labels; detection_coverage reports excluded and unknown checkpoints, not full L0-L3 accuracy.',
        'Physical improvement, risk clearance, public evidence validation and final task success are separate quantities. Temporal associations are not causal attribution.',
        'G1/G0 policy success shares the grounded public completion contract. G1/B0 primary comparison uses common eligibility (original reward plus clean return and no budget/controlled exit, ignoring the extra public phase gate). Raw physical attainment is a separate diagnostic CI, never task success; legacy physical_task_success denotes that raw evaluator flag. Heterogeneous G1/B0 policy completion is descriptive only.',
        'Pending requests retain their reserved upper costs; shared initial requests are billed once. Normal/stress/ablations are separate strata.'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args(argv)
    value=analyze(a.run,n_bootstrap=a.bootstrap)
    with a.output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'episodes':len(value['episodes']),'unique_requests':value['billing']['unique_requests'],'output':str(a.output)}))

if __name__=='__main__':main()
