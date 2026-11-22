"""Cross-campaign initial reuse; no simulator or provider requests."""
import copy,json
from pathlib import Path
import pytest
from capx.experiments import graph_campaign as campaign,graph_v2 as runner
from test_runner_v2 import fake,Env,Budget


def source_run(tmp_path,fake,monkeypatch,kind='valid'):
    # A durable normal campaign with real-shaped provider receipts; no API call.
    config=copy.deepcopy(fake);config.update(services={'sam3':'http://fixture'},study='normal',groups=['B0','G0','G1'],faults=['none'],
        graph_options={'diagnostic_slice':True,'selective_invalidation':True,'selective_recovery':True},
        sources=[str(Path(runner.__file__))],prompt_files=[str(Path(runner.__file__))],scoring_files=[str(Path(runner.__file__))])
    config['allowed_graph_ablations']=[dict(config['graph_options'],diagnostic_slice=False)]
    config['budgets'].update(max_requests=100,cost_limit_usd=100,wall_seconds=10000,max_tokens=32768,reasoning_effort='low',request_timeout=400)
    root=tmp_path/'normal';plan=campaign.make_plan(['stack'],1,1,1)
    campaign.freeze_campaign(root,config,plan,phase='development')
    state=campaign.read(root/'state.json');unit=state['units'][0];unitroot=root/'units'/unit['id'];unitroot.mkdir(parents=True)
    seed=unit['seed'];snapshot={'task':'stack','seed':seed,'state_hash':'same','model_hash':'model'}
    parameters={'model':'deepseek-v4-pro','max_tokens':32768,'reasoning_effort':'low','temperature':.2}
    ledger=[]
    class ReceiptedBudget(Budget):
        def query(self,prompt,label):
            self.count+=1;self.calls.append((prompt,label));dest=root/'requests'/str(self.count)/'llm'/'001-request';dest.mkdir(parents=True)
            response={'content':'```python\nprint("real-shaped")\n```','response_model':'deepseek-v4-pro','finish_reason':'stop','request_parameters':parameters,'response_id':'fixture-'+str(self.count)}
            if kind=='invalid':response['content']='truncated';response['finish_reason']='length'
            receipt={'attempts':1,'requested_model':'deepseek-v4-pro','response_model':'deepseek-v4-pro','max_tokens':32768,'http_status':503 if kind=='api_error' else 200,'status':'error' if kind=='api_error' else 'received','response_id':response['response_id'],'seconds':3.,'usage':{'prompt_tokens':10,'completion_tokens':20},'cost_upper_usd':.1}
            campaign.atomic(dest/'prompt.json',prompt);campaign.atomic(dest/'http.json',{'payload':dict(parameters,messages=prompt)})
            campaign.atomic(dest/'receipt.json',receipt)
            row={'label':unit['id']+':'+label,'prompt_hash':campaign.fingerprint(prompt),'state':'failed' if kind=='api_error' else 'received','directory':str(dest.parent.parent),'response_directory':str(dest),'receipt':receipt,'cost_upper_usd':.1}
            ledger.append(row);campaign.atomic(root/'request-ledger.json',{'requests':ledger})
            if kind=='api_error':raise ConnectionError('known failure')
            campaign.atomic(dest/'response.json',response);return response,str(dest)
    # Use the actual strict parser for invalid cases, restored after source execution.
    if kind=='invalid':
        monkeypatch.setattr(runner,'validated_program',lambda answer: (_ for _ in ()).throw(ValueError('incomplete')))
    result=runner.run_pair(unitroot,'stack',seed,0,snapshot,config,ReceiptedBudget())
    campaign.atomic(unitroot/'pair-result.json',result);unit.update(state='done',result=str(unitroot/'pair-result.json'));campaign.atomic(root/'state.json',state)
    variant=copy.deepcopy(config);variant.update(study='mechanism_ablation',groups=['G0','G1'],graph_options=config['allowed_graph_ablations'][0])
    return root,config,variant,unit,snapshot


def exports(tmp_path,fake,monkeypatch,kind='valid'):
    from capx.experiments.shared_initial import export_initial_manifest
    root,config,variant,unit,snapshot=source_run(tmp_path,fake,monkeypatch,kind)
    target=tmp_path/'initials.json';export_initial_manifest(root,target)
    return root,config,variant,unit,snapshot,target,campaign.read(target)['entries'][0]


def test_valid_reuse_both_graph_arms_no_query_and_same_code(tmp_path,fake,monkeypatch):
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    class NoInitial(Budget):
        def query(self,*args):pytest.fail('cached initial must never call provider')
    b=NoInitial();out=runner.run_pair(tmp_path/'ablation','stack',unit['seed'],0,snapshot,variant,b,groups=('G0','G1'),initial_ref=ref)
    assert b.count==0 and len(out['episodes'])==2
    assert all(r['new_llm_requests']==0 and r['llm_requests']==1 and r['initial_reused'] for r in out['episodes'])
    assert out['episodes'][0]['code_hash']==out['episodes'][1]['code_hash']==campaign.read(Path(ref['initial_file']))['code_hash']
    assert all(r['shared_initial_seconds_charged']>=0 for r in out['episodes'])
    assert len(campaign.read(root/'request-ledger.json')['requests'])==2


@pytest.mark.parametrize('kind',['invalid','api_error'])
def test_failed_initial_is_exported_and_shared_without_success_filter(tmp_path,fake,monkeypatch,kind):
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch,kind)
    b=Budget();b.query=lambda *a:pytest.fail('no failed-initial retry')
    out=runner.run_pair(tmp_path/'ablation','stack',unit['seed'],0,snapshot,variant,b,groups=('G0','G1'),initial_ref=ref)
    assert len(campaign.read(path)['entries'])==1
    assert all(r['reward']==0 and r['stop_reason']=='initial_generation_failed' and r['new_llm_requests']==0 for r in out['episodes'])
    assert all(r['llm_requests']==1 for r in out['episodes'])


@pytest.mark.parametrize('field',['snapshot','identity','fault','model','max_tokens','reasoning_effort','source','receipt','response','initial','missing'])
def test_reference_drift_fails_closed_without_query(tmp_path,fake,monkeypatch,field):
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    args={'task':'stack','seed':unit['seed'],'repeat':0,'snapshot':snapshot,'config':variant,'fault':'none'}
    if field=='snapshot':args['snapshot']=dict(snapshot,state_hash='other')
    elif field=='identity':args['repeat']=1
    elif field=='fault':args['fault']='target_shift'
    elif field=='model':variant['model']='other'
    elif field in ('max_tokens','reasoning_effort'):variant['budgets'][field]=123 if field=='max_tokens' else 'high'
    elif field=='source':ref['source_signature']='corrupted'
    else:
        p=Path(ref['initial_file'] if field in ('initial','missing') else ref['request_directory'])
        if field not in ('initial','missing'):p=p/(field+'.json')
        if field=='missing':p.unlink()
        else:p.write_text('{}')
    b=Budget();b.query=lambda *a:pytest.fail('must fail closed')
    with pytest.raises((ValueError,FileNotFoundError)):
        runner.run_pair(tmp_path/'bad',request_budget=b,groups=('G0','G1'),initial_ref=ref,**args)
    assert b.count==0


def test_export_requires_all_fixed_units_and_refuses_uncertain_request(tmp_path,fake,monkeypatch):
    from capx.experiments.shared_initial import export_initial_manifest
    root,config,variant,unit,snapshot=source_run(tmp_path,fake,monkeypatch)
    ledger=campaign.read(root/'request-ledger.json');ledger['requests'][-1]['state']='uncertain';campaign.atomic(root/'request-ledger.json',ledger)
    with pytest.raises(ValueError,match='uncertain'):export_initial_manifest(root,tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()


def test_manifest_matches_exact_allocation_and_is_frozen(tmp_path,fake,monkeypatch):
    from capx.experiments.shared_initial import validate_initial_manifest
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    target=dict(unit,groups=['G0','G1'])
    indexed=validate_initial_manifest(path,variant,[target],phase='development')
    assert indexed[('stack',unit['seed'],0,'none')]['source_signature']==ref['source_signature']
    with pytest.raises(ValueError,match='allocation'):validate_initial_manifest(path,variant,[dict(target,repeat=1)],phase='development')
    with pytest.raises(ValueError,match='formal'):validate_initial_manifest(path,variant,[target],phase='formal')
    variant['initial_manifest']=str(path)
    inventory=campaign.source_inventory(variant)
    assert str(path.resolve()) in inventory and str(Path(ref['initial_file']).resolve()) in inventory


def test_cli_export_and_pair_option_present():
    import subprocess,sys
    for module,argument in [('capx.experiments.graph_campaign','export-initials'),('capx.experiments.graph_v2',None)]:
        args=[sys.executable,'-B','-m',module]+([argument] if argument else [])+['--help']
        r=subprocess.run(args,capture_output=True,text=True);assert r.returncode==0,r.stderr
        if argument is None:assert '--initial-manifest' in r.stdout


def test_prompt_and_docs_are_rechecked_after_new_environment_reset(tmp_path,fake,monkeypatch):
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    monkeypatch.setattr(runner,'initial_prompt',lambda base,obs:[{'role':'user','content':'changed public state'}])
    b=Budget();b.query=lambda *a:pytest.fail('pair mismatch must not regenerate')
    out=runner.run_pair(tmp_path/'ablation','stack',unit['seed'],0,snapshot,variant,b,groups=('G0','G1'),initial_ref=ref)
    assert all(r['status']=='invalid_program' and r['actual_api_calls']==0 for r in out['episodes'])
    assert all('Paired initial_prompt_hash drift' in r['error'] for r in out['episodes'])


def test_incomplete_allocation_cannot_export_only_successes(tmp_path,fake,monkeypatch):
    from capx.experiments.shared_initial import export_initial_manifest
    root,config,variant,unit,snapshot=source_run(tmp_path,fake,monkeypatch)
    state=campaign.read(root/'state.json');state['units'][0]['state']='pending';campaign.atomic(root/'state.json',state)
    with pytest.raises(ValueError,match='complete fixed allocation'):export_initial_manifest(root,tmp_path/'bad.json')


def test_manifest_duplicates_and_source_phase_tampering_rejected(tmp_path,fake,monkeypatch):
    from capx.experiments.shared_initial import validate_initial_manifest
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    body=campaign.read(path);body['entries'].append(ref);duplicate=tmp_path/'duplicates.json';campaign.atomic(duplicate,body)
    with pytest.raises(ValueError,match='Duplicate'):validate_initial_manifest(duplicate,variant)
    body['entries'].pop();body['source_phase']='formal';campaign.atomic(duplicate,body)
    with pytest.raises(ValueError,match='source mismatch'):validate_initial_manifest(duplicate,variant,phase='formal')


def test_reference_is_archived_and_resume_rejects_drift(tmp_path,fake,monkeypatch):
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    variant['initial_manifest']=str(path);target=tmp_path/'target';plan=campaign.read(root/'plan.json')
    campaign.freeze_campaign(target,variant,plan,phase='development')
    frozen=campaign.read(target/'frozen.json')
    assert str(path.resolve()) in frozen['source_archive']
    assert str(Path(ref['initial_file']).resolve()) in frozen['source_archive']
    Path(ref['request_directory'],'receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='drift'):campaign.load_campaign(target)


def test_cross_root_billing_reuses_canonical_receipt(tmp_path,fake,monkeypatch):
    from capx.experiments.graph_v2_analysis import billing
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    out=runner.run_pair(tmp_path/'ablation','stack',unit['seed'],0,snapshot,variant,Budget(),groups=('G0','G1'),initial_ref=ref)
    source_rows=campaign.read(Path(unit['result']))['episodes']
    result=billing([root,tmp_path/'ablation'],source_rows+out['episodes'])
    assert result['unique_requests']==2 and result['cost_upper_usd']==pytest.approx(.2)


def test_b0_cannot_use_grounded_initial(tmp_path,fake,monkeypatch):
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    b=Budget();b.query=lambda *a:pytest.fail('wrong interface must not request')
    with pytest.raises(ValueError,match='B0'):
        runner.run_pair(tmp_path/'bad','stack',unit['seed'],0,snapshot,variant,b,groups=('B0','G0','G1'),initial_ref=ref)


def test_campaign_passes_exact_reference_without_adding_request(tmp_path,fake,monkeypatch):
    root,config,variant,unit,snapshot,path,ref=exports(tmp_path,fake,monkeypatch)
    variant['initial_manifest']=str(path);target=tmp_path/'target';plan=campaign.read(root/'plan.json')
    campaign.freeze_campaign(target,variant,plan,phase='development')
    calls=[]
    def pair(**kw):
        calls.append(kw['initial_ref'])
        return {'episodes':[dict(task=kw['task'],seed=kw['seed'],repeat=kw['repeat'],fault=kw['fault'],group=g,status='task_failure',reward=0) for g in kw['groups']]}
    result=campaign.run_campaign(target,runner=pair)
    assert len(calls)==1 and calls[0]==ref and result['requests']==0
    assert campaign.run_campaign(target,runner=lambda **kw:pytest.fail('no repeat'))['complete_units']==1
