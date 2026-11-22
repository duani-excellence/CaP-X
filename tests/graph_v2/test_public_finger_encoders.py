"""Both finger encoders are deployable proprioception, never object/contact truth."""
import numpy as np
from capx.execution.feedback import public_observation
from capx.experiments.vision_tasks import public_sensors


def test_both_public_finger_encoders_survive_boundary_copy():
    raw={'robot_joint_pos':np.zeros(8), 'robot_cartesian_pos':np.zeros(8),
         'robot_gripper_joint_pos':np.array([.031,-.019]),
         'cube_poses':{'hidden':True},'contact_forces':np.ones(3)}
    for clean in (public_observation(raw),public_sensors(raw,[])):
        assert np.allclose(clean['robot_gripper_joint_pos'],[.031,-.019])
        clean['robot_gripper_joint_pos'][0]=0
        assert raw['robot_gripper_joint_pos'][0]==.031
        assert 'cube_poses' not in clean and 'contact_forces' not in clean


def test_optional_encoders_do_not_manufacture_measurement():
    assert 'robot_gripper_joint_pos' not in public_observation({})
    assert 'robot_gripper_joint_pos' not in public_sensors({},[])
