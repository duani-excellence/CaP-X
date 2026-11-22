from capx.experiments.graph_v2_analysis import active_evidence_attempts,intervention_records


def test_semantic_refresh_is_evidence_acquisition_not_physical_repair():
    event={'layer':'L0','check':'semantic_refinement','status':'PASS',
        'metrics':{'attempted':True,'physical_motion':False,
                   'initial_validation':{'status':'UNCERTAIN'},'final_validation':{'status':'PASS'}}}
    recovery=active_evidence_attempts([(0,event)],[],{},[])
    assert len(recovery)==1
    row=intervention_records(recovery,{'reward':1,'policy_success':True,'replans':0})[0]
    assert row['kind']=='semantic_refinement' and row['role']=='evidence_acquisition'
    assert row['motion_calls']==0 and row['detector_cleared'] is True
    assert row['physical_risk_cleared'] is None and not row['independent_physical_repair_evidence']
