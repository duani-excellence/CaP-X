"""Public F/T mechanics. These tests do not assert simulator task success."""
import numpy as np
import pytest
from capx.integrations.sensors.wrist_ft import transform_wrench, compensate_gravity, ForceWindow


def test_wrench_shift_rotates_force_and_translates_moment():
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    out=transform_wrench([1,0,0,0,0,0],r,[0,0,.1])
    assert np.allclose(out,[0,1,0,-.1,0,0])


def test_empty_tool_compensation_keeps_unknown_payload_load():
    tool=np.array([0,0,5.,0,.2,0]);payload=np.array([0,0,1.,0,.1,0])
    # MuJoCo internal reaction = minus external gravity wrench.
    assert np.allclose(compensate_gravity(-tool,tool),0)
    assert np.allclose(compensate_gravity(-tool-payload,tool),payload)


def packet(t,w=(0,0,-1,0,0,0),**extra):
    return dict(timestamp=t,valid=True,quasistatic=True,external_base=list(w),
                gripper_width_m=.04,tcp=[0,0,0,1,0,0,0],**extra)


def test_force_window_rejects_stale_dynamic_and_insufficient_evidence():
    m=ForceWindow();m.append(packet(1))
    assert m.assess(1)['status']=='UNKNOWN'
    for i in range(1,9):m.append(packet(1+i*.05))
    assert m.assess(1.4)['status']=='PASS'
    assert m.assess(2)['status']=='UNKNOWN'
    x=packet(1.45);x['quasistatic']=False;m.append(x)
    assert m.assess(1.45)['status']=='UNKNOWN'


def test_force_window_load_is_not_force_closure_or_release_permission():
    m=ForceWindow()
    for i in range(10):m.append(packet(i*.05))
    x=m.assess(.45)
    assert x['load_n']>0 and x['hold_permission'] is False
    assert x['release_permission'] is False


def test_load_loss_and_jaw_motion_are_not_stable_grasp():
    m=ForceWindow()
    for i in range(10):
        x=packet(i*.05);x['gripper_width_m']=.04-i*.002;m.append(x)
    assert m.assess(.45)['status']=='UNKNOWN'
    m=ForceWindow()
    for i in range(10):m.append(packet(i*.05,w=(0,0,0,0,0,0)))
    assert m.assess(.45)['status']=='UNKNOWN'


def test_malformed_and_backwards_sensor_samples_fail_closed():
    m=ForceWindow();m.append(packet(1))
    with pytest.raises(ValueError):m.append(packet(.9))
    x=packet(2);x['external_base'][0]=float('nan')
    with pytest.raises(ValueError):m.append(x)


def test_hardware_adapter_refuses_guessed_identity_and_units():
    from capx.integrations.sensors.wrist_ft import ConfiguredHardwareFT
    calls=[]
    with pytest.raises(ValueError):ConfiguredHardwareFT(lambda:calls.append(1),lambda x,c:x,{})
    assert calls==[]


def force_skills(widths,assessment):
    from types import SimpleNamespace
    from capx.execution.force_skills import ForceRegionSkills
    s=ForceRegionSkills.__new__(ForceRegionSkills);s.force_graph=None;s.force_decisions=[];s.load_anchors={};s.emit=lambda *a,**k:None
    s.held={'object':'renamed','tcp':np.eye(4)};s.active_condition='held:renamed';s.effects={};s.phase='closed_unverified'
    s.scene=SimpleNamespace(source='public',frame=lambda:None,obs={'robot_gripper_joint_pos':[.024,-.024],'wrist_ft':{'valid':True}})
    it=iter(widths);calls=[]
    def motion(name,*args):
        calls.append((name,args));v=next(it);s.scene.obs['robot_gripper_joint_pos']=[v/2,-v/2]
    s.motion=motion;s.force_evidence=lambda:assessment
    return s,calls


def test_force_guided_settle_extends_only_with_actual_progress():
    s,calls=force_skills([.045,.042,.040,.039,.0385,.0382],{'status':'PASS','load_n':1.})
    s._settle_attachment('renamed')
    assert len(calls)>3 and len(calls)<=20
    assert s.phase=='closed_unverified' and s.effects=={}  # still requires co-motion
    assert all(name=='wait' for name,_ in calls)


def test_force_guided_missing_load_never_claims_attachment():
    from capx.execution.skill_contracts import SkillBoundaryStop
    s,calls=force_skills([.045]*20,{'status':'UNKNOWN','load_n':0.})
    with pytest.raises(SkillBoundaryStop,match='load missing'):s._settle_attachment('renamed')
    assert len(calls)==3 and s.effects['held:renamed']=='UNKNOWN'


def test_no_progress_cannot_wait_forever():
    from capx.execution.skill_contracts import SkillBoundaryStop
    s,calls=force_skills([.045]*25,{'status':'UNKNOWN'})
    with pytest.raises(SkillBoundaryStop,match='No progress'):s._settle_attachment('renamed')
    assert len(calls)<20


def test_four_way_cli_shares_skills_and_strictly_gates_modes():
    from capx.experiments.unified import RunSpec,resolve
    s,c,f=resolve(RunSpec(force_mode='compare'))
    assert s.skill=='v32' and c['tasks']['pick_place']['force_sensor'] and f=='contract'
    with pytest.raises(ValueError):resolve(RunSpec(force_mode='on',task='stack'))
    with pytest.raises(ValueError):resolve(RunSpec(force_mode='on',skill='v30'))


def test_public_capture_never_exposes_force_in_disabled_arm(tmp_path):
    from types import SimpleNamespace
    from capx.experiments.native_tasks import NativeBoundary
    b=NativeBoundary.__new__(NativeBoundary);b.root=tmp_path;b.captures=0;b.force_enabled=False
    b.low=SimpleNamespace(render_camera_names=[],robosuite_env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=1))))
    b.public=SimpleNamespace(get_observation=lambda:{'robot_joint_pos':[0]*7})
    b.force_sensor=SimpleNamespace(sample=lambda:{'valid':True,'source':'wrist'});b.log=lambda *a,**k:None
    assert 'wrist_ft' not in b.capture()
    b.force_enabled=True;assert b.capture()['wrist_ft']['source']=='wrist'


def test_calibrated_resolution_distinguishes_small_load_from_uncertainty():
    m=ForceWindow(force_error_bound_n=.005)
    for i in range(10):m.append(packet(i*.05,w=(0,0,-.03,0,0,0)))
    assert m.assess(.45)['status']=='PASS'
    assert m.assess(.45)['load_interval_n']==pytest.approx([.025,.035])
    uncertain=ForceWindow(force_error_bound_n=.04)
    for i in range(10):uncertain.append(packet(i*.05,w=(0,0,-.03,0,0,0)))
    assert uncertain.assess(.45)['status']=='UNKNOWN'


def test_observed_attachment_template_preserves_frame_and_rejects_flat_surface():
    from capx.execution.force_skills import observed_attachment_template
    rng=np.random.default_rng(8);cloud=rng.normal(size=(1000,3))*.012
    t=np.eye(4);t[:3,3]=[.4,.2,.1]
    template=observed_attachment_template(cloud,t)
    assert np.allclose(template@t[:3,:3].T+t[:3,3],cloud)
    with pytest.raises(ValueError):observed_attachment_template(np.c_[cloud[:,:2],np.zeros(1000)],t)
    with pytest.raises(ValueError):observed_attachment_template(cloud[:50],t)


def test_observed_template_still_rejects_actual_rotation():
    from capx.execution.force_skills import observed_attachment_template
    from capx.execution.skill_recovery import attachment_consistency
    from scipy.spatial.transform import Rotation
    t=np.eye(4);t1=t.copy();t1[2,3]=.01
    bad=t1.copy();bad[:3,:3]=Rotation.from_euler('z',10,degrees=True).as_matrix()
    assert attachment_consistency(t,t1,t,bad,.04,.04)['status']=='UNKNOWN'


def test_force_graph_expires_old_window_without_changing_visual_hold():
    from capx.experiments.native_vision_skills import region_task
    from capx.execution.skill_contracts import ContractGraph
    s,_=force_skills([],{'status':'PASS','load_n':1.})
    c=ContractGraph(region_task({'renamed':'part'},grounded=True,causal=True));s.force_graph=c
    c.observe('held:renamed','TRUE',{'source':'independent_current_RGBD'})
    s._force_record('force_load',dict(status='PASS',timestamp=1.,expires_at=1.15))
    old=c.conditions['force_load:renamed']
    s._force_record('force_contact',dict(status='UNKNOWN',timestamp=1.3))
    assert c.truth('force_load:renamed')=='UNKNOWN' and not c.graph.nodes[old]['valid']
    assert c.truth('held:renamed')=='TRUE'  # force window is not the visual certificate


def test_load_interval_includes_observed_window_variation():
    m=ForceWindow(force_error_bound_n=.005)
    for i in range(10):m.append(packet(i*.05,w=(0,0,-(.03+(.02 if i%2 else -.02)),0,0,0)))
    e=m.assess(.45)
    assert e['load_interval_n'][0]<=.005 and e['load_interval_n'][1]>=.055
    assert e['status']=='UNKNOWN'
