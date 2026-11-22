"""Direct scene-paired full-G1 versus one predeclared ablated-G1 mechanism.

No new estimator: alias the two G1 arms only inside the existing paired cluster
bootstrap. Source group labels and allocation provenance remain explicit.
"""
from __future__ import annotations
from pathlib import Path


def _allocation(root,frozen):
    from capx.experiments.graph_campaign import read,validate_checkpoint,_validate_result
    from capx.experiments.shared_initial import identity
    state=read(root/'state.json');validate_checkpoint(frozen,state)
    ledger=read(root/'request-ledger.json') if (root/'request-ledger.json').exists() else {'requests':[]}
    result={}
    for unit in state['units']:
        if unit['state']!='done':continue
        rows=_validate_result(read(unit['result']),unit)
        result[identity(unit)]={r['group']:r for r in rows}
    unknown=sum(r.get('state') in ('started','uncertain') for r in ledger['requests'])
    report=dict(planned_units=len(state['units']),complete_units=len(result),uncertain_requests=unknown,
                incomplete_units=[u['id'] for u in state['units'] if u['state']!='done'])
    report['source_integrity_complete']=not any(r.get('status')=='source_changed' or r.get('source_changes') or (r.get('source_postcheck') or {}).get('unchanged') is False for groups in result.values() for r in groups.values())
    report['complete']=report['planned_units']==report['complete_units'] and not unknown and report['source_integrity_complete']
    return state,result,report


def paired_mechanism_contrast(normal_root,ablation_root,*,n_bootstrap=2000,random_seed=0):
    """Compare the fixed subset; missing/failed outcomes are never selected away.

    Both campaigns must be formal, compatible and sourced from the same frozen
    normal initial allocation. Incomplete target outcomes yield descriptive
    available-pair estimates only, with exact missing identities retained.
    """
    from capx.experiments.graph_campaign import (load_campaign,compatible_development,
        declared_sources,fingerprint,planned_units)
    from capx.experiments.shared_initial import validate_initial_manifest,identity
    from capx.experiments.graph_statistics import paired_summary
    normal_root=Path(normal_root).resolve();ablation_root=Path(ablation_root).resolve()
    if normal_root==ablation_root:raise ValueError('Distinct normal and ablation campaigns required')
    normal=load_campaign(normal_root);ablated=load_campaign(ablation_root)
    nc,ac=normal['config'],ablated['config']
    if normal['phase']!='formal' or ablated['phase']!='formal':raise ValueError('Mechanism contrast requires formal frozen campaigns')
    if nc.get('study','normal')!='normal' or ac.get('study')!='mechanism_ablation':raise ValueError('Normal versus mechanism_ablation required')
    if nc.get('faults',['none'])!=['none'] or ac.get('faults',['none'])!=['none']:raise ValueError('Mechanism contrast excludes stress faults')
    if set(nc.get('groups',[]))!={'B0','G0','G1'} or set(ac.get('groups',[]))!={'G0','G1'}:raise ValueError('Unexpected mechanism groups')
    declaration=compatible_development(ac,nc)
    if not declaration:raise ValueError('Exactly one predeclared graph mechanism must differ')
    if nc['tasks']!=ac['tasks'] or fingerprint(normal['plan'])!=fingerprint(ablated['plan']):raise ValueError('Different complete allocation plans/tasks')
    if declared_sources(nc)!=declared_sources(ac):raise ValueError('Different business source set')
    if not ac.get('initial_manifest'):raise ValueError('Shared initial manifest required')
    ns,nrows,nstatus=_allocation(normal_root,normal);ats,arows,astatus=_allocation(ablation_root,ablated)
    entries=validate_initial_manifest(ac['initial_manifest'],ac,ats['units'],phase='formal',plan=ablated['plan'])
    expected={identity(u):u for u in planned_units(ac,ablated['plan'],'formal')}
    if set(entries)!=set(expected):raise ValueError('Initial source selection differs from predeclared subset')
    pairs=[];four_hashes=('snapshot_hash','initial_prompt_hash','api_docs_hash','code_hash')
    for key,entry in entries.items():
        if Path(entry['source_campaign']).resolve()!=normal_root or entry['source_signature']!=normal['signature']:
            raise ValueError('Initial source campaign/signature differs from selected normal campaign')
        # Copies use the established G1/G0 bootstrap interface; physical runs
        # remain G1 in both roots. This is not the within-campaign G1-G0 effect.
        for root,rows,alias in ((normal_root,nrows,'G1'),(ablation_root,arows,'G0')):
            if key not in rows:continue
            row=rows[key]['G1']
            if row.get('fault','none')!='none':raise ValueError('Stress row mixed into mechanism contrast')
            if alias=='G0':
                origin=row.get('initial_source') or {}
                if row.get('initial_reused') is not True:raise ValueError('Ablated G1 did not reuse its normal initial')
                for field in ('source_campaign','source_signature','source_unit_id','source_request_uid'):
                    if origin.get(field)!=entry.get(field) or not origin.get(field):raise ValueError('Ablated initial source mismatch: '+field)
                if origin.get('generated_in_this_run') is not False or origin.get('initial_requests_this_run')!=0:
                    raise ValueError('Ablated initial unexpectedly regenerated')
            if not all(row.get(field) for field in four_hashes):raise ValueError('Missing shared initial pair hash')
            pairs.append(dict(row,group=alias,ablation='direct_mechanism_contrast',
                              source_group='G1',source_ablation=row.get('ablation','full'),source_campaign=str(root)))
    # Ensure even a completely missing scene (both arms) is explicit, rather
    # than vanishing from available-pair output. These are unprepared markers,
    # not fabricated task failures, and force an incomplete estimate.
    present={(identity(r),r['group']) for r in pairs}
    for key,unit in expected.items():
        for alias in ('G1','G0'):
            if (key,alias) not in present:
                pairs.append(dict(task=key[0],seed=key[1],repeat=key[2],fault=key[3],group=alias,
                                  ablation='direct_mechanism_contrast',executed=False,status='scene_preparation_error',
                                  reward=None,missing_campaign_outcome=True,source_group='G1'))
    options=dict(n_bootstrap=n_bootstrap,random_seed=random_seed,expected_repeats=normal['plan']['repeats'])
    policy=paired_summary(pairs,'G1','G0',**options)
    physical=paired_summary(pairs,'G1','G0',outcome='original_physical_attainment',**options)
    complete=nstatus['complete'] and astatus['complete']
    for summary in (policy,physical):
        summary.update(treatment='full_G1',control='ablated_G1',formal_estimate_ready=summary['formal_estimate_ready'] and complete)
    return dict(schema='capx.direct_mechanism_contrast.v1',normal_campaign=str(normal_root),ablation_campaign=str(ablation_root),
        normal_signature=normal['signature'],ablation_signature=ablated['signature'],declaration=declaration,
        normal_status=nstatus,ablation_status=astatus,formal_estimate_ready=policy['formal_estimate_ready'],
        selection=dict(rule=ac.get('mechanism_scene_subset') or 'complete formal allocation',normal_units=len(ns['units']),
            selected_units=len(expected),repeats=normal['plan']['repeats'],outcome_filtering=False),
        policy_contrast=policy,physical_attainment_contrast=physical,
        definitions=dict(effect='Full-G1 minus ablated-G1 on the same predeclared scenes, Snapshot and initial program; both use identical grounded completion policy',
            clustering='Existing equal-task mean difference; resample whole (task, seed) clusters with all model draws paired',
            missing='Incomplete/unknown campaign units prevent formal readiness; all attempted failures retained; missing markers are never task failures',
            physical='Original physical state attainment diagnostic, not task success; controlled/budget exits remain policy failures',
            multiplicity='Each of three predeclared secondary mechanism contrasts has its own unadjusted 95% CI, not a simultaneous familywise CI',
            cost='Paired resource deltas charge the shared initial logically in both arms; actual billing must deduplicate canonical request receipts across campaign roots'))
