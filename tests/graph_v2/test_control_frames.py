"""Robot calibration is static metadata; no scene truth or inferred seed offsets."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

MJCF = '''<mujoco><worldbody><body name="robot0_link0" pos="1 2 3"><body name="fixed_mount0_base" pos="0 0 .01"/><body name="robot0_link7"><joint name="joint7"/><body name="robot0_right_hand" pos="0 0 .1065" quat=".924 0 0 -.383"/></body></body></worldbody></mujoco>'''
URDF = '''<robot name="panda"><link name="panda_link0"/><link name="panda_link7"/><link name="panda_link8"/><link name="panda_hand"/><joint name="panda_joint8" type="fixed"><parent link="panda_link7"/><child link="panda_link8"/><origin xyz="0 0 .107" rpy="0 0 0"/></joint><joint name="panda_hand_joint" type="fixed"><parent link="panda_link8"/><child link="panda_hand"/><origin xyz="0 0 0" rpy="0 0 -0.7853981633974483"/></joint></robot>'''

def make_cal(xml=MJCF, urdf=URDF):
    from capx.execution.control_frames import calibration_from_xml
    return calibration_from_xml(xml, urdf, public_base='fixed_mount0_base', actual_base='robot0_link0', actual_parent='robot0_link7', actual_hand='robot0_right_hand', actual_tcp_offset=[0,0,.107], api_tcp_to_hand_offset=[0,0,-.107])

def pose(p,q):
    result=np.eye(4);result[:3,:3]=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix();result[:3,3]=p
    return result

def shift(p):
    out=np.eye(4);out[:3,3]=p;return out

@pytest.mark.parametrize('q', [[1,0,0,0],[0,0,1,0],[.5,.5,.5,.5]])
def test_mapping_reconstructs_same_desired_physical_tcp_for_arbitrary_orientation(q):
    from capx.execution.control_frames import transform_target
    cal=make_cal(); p=[.42,.08,-.06]
    mp,mq=transform_target(cal,p,q)
    # Original goto applies negative offset. Reconstruct actual hand / public TCP.
    ik_hand=pose(mp,mq) @ shift(cal['api_tcp_to_hand_offset_m'])
    public_tcp=np.linalg.inv(cal['T_ik_base_public_base']) @ ik_hand @ np.linalg.inv(cal['T_actual_hand_ik_hand']) @ shift(cal['actual_tcp_offset_in_hand_m'])
    assert public_tcp==pytest.approx(pose(p,q),abs=1e-12)
    assert np.linalg.norm(mq)==pytest.approx(1)

def test_sign_matches_source_defined_mount_and_hand_offsets_without_empirical_bias():
    from capx.execution.control_frames import transform_target
    cal=make_cal();p,q=transform_target(cal,[.5,0,.2],[0,0,1,0])
    assert p==pytest.approx([.5,0,.2095],abs=1e-12)
    assert cal['T_ik_base_public_base'][2][3]==pytest.approx(.01)
    # Changing static calibration must change target; no fixed .0095 compensation.
    changed=make_cal(MJCF.replace('0 0 .01','0 0 .03'))
    assert transform_target(changed,[.5,0,.2],[0,0,1,0])[0][2]==pytest.approx(.2295)

def test_hash_is_deterministic_json_safe_and_rejects_tampering():
    from capx.execution.control_frames import transform_target
    cal=make_cal();assert json.loads(json.dumps(cal))==cal
    assert make_cal()['calibration_hash']==cal['calibration_hash']
    bad=copy.deepcopy(cal);bad['T_ik_base_public_base'][2][3]=.1
    with pytest.raises(ValueError,match='hash'):transform_target(bad,[0,0,0],[1,0,0,0])

@pytest.mark.parametrize('field,value', [('position',[1,2,float('nan')]),('quaternion',[0,0,0,0]),('quaternion',[1,2,3])])
def test_invalid_inputs_fail_closed(field,value):
    from capx.execution.control_frames import transform_target
    p,q=[0,0,0],[1,0,0,0]
    if field=='position':p=value
    else:q=value
    with pytest.raises(ValueError):transform_target(make_cal(),p,q)

def test_missing_or_movable_static_chain_fails_closed():
    with pytest.raises(ValueError):make_cal(MJCF.replace('name="fixed_mount0_base"','name="wrong"'))
    with pytest.raises(ValueError):make_cal(MJCF.replace('pos="0 0 .1065"','pos="0 0 .1065" euler="0 0 0"'))
    with pytest.raises(ValueError):make_cal(urdf=URDF.replace('type="fixed"','type="revolute"',1))
    with pytest.raises(ValueError):make_cal(MJCF.replace('pos="0 0 .01"/>','pos="0 0 .01"><joint name="movable"/></body>'))

def test_adapter_extracts_only_static_robot_model_and_preserves_old_api(tmp_path,monkeypatch):
    from capx.experiments import vision_tasks
    from capx.utils import robot_urdf
    path=tmp_path/'panda.urdf';path.write_text(URDF)
    source=tmp_path/'robot.xml';source.write_text(MJCF)
    mount=tmp_path/'mount.xml';mount.write_text('<mujoco/>')
    gripper=tmp_path/'gripper.xml';gripper.write_text('<mujoco model="configured_gripper"/>')
    model=SimpleNamespace(naming_prefix='robot0_',root_body='robot0_link0',get_xml=lambda:MJCF,file=str(source),base=SimpleNamespace(root_body='fixed_mount0_base',file=str(mount)))
    class NeverPhysics:
        def __getattr__(self,name):raise AssertionError('Physics access forbidden: '+name)
    low=SimpleNamespace(robosuite_env=SimpleNamespace(robots=[SimpleNamespace(robot_model=model,gripper={'right':SimpleNamespace(file=str(gripper))})],sim=NeverPhysics()))
    original=lambda *a:None
    api=SimpleNamespace(_TCP_OFFSET=np.array([0,0,-.107]),goto_pose=original)
    env=SimpleNamespace(low_level_env=low,_apis={'FrankaControlApi':api})
    monkeypatch.setattr(robot_urdf,'resolve_robot_urdf_path',lambda name:str(path))
    cal=vision_tasks.public_robot_calibration(env)
    assert cal['frames']['public_base']=='fixed_mount0_base'
    assert {r['role'] for r in cal['sources']} >= {'robot_mjcf','mount_mjcf','ik_urdf','gripper_mjcf'}
    configured=next(r for r in cal['sources'] if r['role']=='gripper_mjcf')
    assert configured['path']==str(gripper.resolve())
    import hashlib
    assert configured['sha256']==hashlib.sha256(gripper.read_bytes()).hexdigest()
    assert api.goto_pose is original
    assert not any(k in cal for k in ['objects','canonical','contacts','success'])
    monkeypatch.setattr(robot_urdf,'resolve_robot_urdf_path',lambda name:'uninstalled_description')
    with pytest.raises(FileNotFoundError):vision_tasks.public_robot_calibration(env)


def test_mjcf_real_layout_has_link0_and_mount_as_fixed_siblings():
    # Real Panda's robot_model.root_body is robot0_base, not robot0_link0.
    nested=MJCF.replace('<body name="robot0_link0" pos="1 2 3">', '<body name="robot0_base" pos="1 2 3"><body name="robot0_link0"/>')
    assert make_cal(nested)['T_ik_base_public_base'][2][3]==pytest.approx(.01)
