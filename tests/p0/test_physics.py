from capx.envs.physical_evaluation import PhysicalCriteria

def sample(t, *, z=0.82, contact=False, success=False, speed=0., opening=.08):
    return dict(sim_time=t, objects={'primary': {'position': [0,0,z], 'speed': speed}},
                grasp_contact=contact, raw_success=success, gripper_opening=opening)

def test_grasp_slip_and_placement():
    c = PhysicalCriteria(sample(0))
    c.update(sample(.1,z=.9,contact=True,opening=.03))
    c.update(sample(.7,z=.9,contact=True,opening=.03))
    assert c.grasp_success
    c.update(sample(.8,z=.82))
    assert c.slipped
    c.update(sample(1,success=True))
    assert not c.stable_placement
    c.update(sample(1.6,success=True))
    assert c.stable_placement

def test_contact_or_instant_success_not_enough():
    c = PhysicalCriteria(sample(0))
    c.update(sample(1,contact=True,success=True,opening=.03))
    assert not c.grasp_success and not c.stable_placement
    c.update(sample(2,success=True,speed=.5))
    assert not c.stable_placement

def test_intentional_release_not_slip():
    c=PhysicalCriteria(sample(0))
    c.update(sample(.1,z=.9,contact=True,opening=.03))
    c.update(sample(.7,z=.9,contact=True,opening=.03))
    release=sample(.8,z=.9,opening=.03)
    release['gripper_command']=1.0
    c.update(release)
    assert not c.slipped
