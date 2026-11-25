from types import SimpleNamespace
from capx.execution.decision_state import encode_decision_state
import numpy as np


def test_initial_checkpoint_does_not_invent_uncaptured_robot_state():
    s=SimpleNamespace(held=None,phase='empty',scene=SimpleNamespace())
    d=encode_decision_state({},s,{},[],{},[])
    assert d['robot']['tcp'] is None and d['robot']['source'] is None
    assert d['interaction']['force_observation'] is None


def test_decision_separates_kinematics_path_release_and_search_range():
    s=SimpleNamespace(held=None,phase='empty',scene=SimpleNamespace(obs={},source='public'))
    searches=[dict(id='s1',status='NO_SOLUTION_FOUND',tested=[],candidate_count=48,remaining_candidates=0,path_status='NOT_VERIFIED'),
              dict(id='s2',status='KINEMATIC_CANDIDATE',tested=[],candidate_count=1,remaining_candidates=0,path_status='NOT_VERIFIED')]
    d=encode_decision_state({'remaining_goals':[]},s,{},searches,{},[])
    assert d['plans'][0]['decision_status']=='NO_SOLUTION_IN_TESTED_SET'
    assert d['plans'][1]['decision_status']=='IK_FEASIBLE'
    assert d['plans'][1]['release_status']=='RELEASE_UNVERIFIED'
    assert d['coordinates']['quaternion_order']=='WXYZ'
    assert d['planning_statistics']['unique_searches']==2


def test_force_evidence_is_not_attachment_pose_and_quality_is_separate():
    s=SimpleNamespace(held=None,phase='empty',scene=SimpleNamespace(obs={'wrist_ft':{'valid':True,'wrench':[0]*6}},source='f'))
    b={'quality:novel':{'values':{'boundary_margin_m':.02},'source':'q'}}
    d=encode_decision_state({'remaining_goals':['placed:novel']},s,b,[],{},[])
    assert 'wrench' in d['interaction']['force_observation']
    assert 'not' in d['interaction']['interpretation']
    assert d['placement_quality']['novel']['source']=='q'


def test_attachment_name_is_consistent_with_execution_state():
    held={'object':'novel','tcp':np.eye(4),'shape':{'center':[.4,.1,.2]}}
    s=SimpleNamespace(held=held,phase='closed_unverified',scene=SimpleNamespace(obs={},source='f'))
    state={'attachment':'novel','conditions':{'held:novel':{'truth':'UNKNOWN'}}}
    d=encode_decision_state(state,s,{},[],{},[])
    assert d['attachment']==state['attachment']
    assert d['attachment_estimate']['object']=='novel'
    assert d['attachment_estimate']['current_status']=='UNKNOWN'
    s.held=None
    d=encode_decision_state({'attachment':None},s,{},[],{},[])
    assert d['attachment'] is None and d['attachment_estimate'] is None
