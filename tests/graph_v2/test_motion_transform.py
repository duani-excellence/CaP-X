import numpy as np
from capx.execution.grounded_feedback import GroundedMonitor


def test_only_actual_robot_motion_gets_frame_mapping():
    state={'robot_cartesian_pos':np.array([.5,0,.2,0,0,1,0,.5]),'robot_joint_pos':np.zeros(8)}
    transforms=[];calls=[]
    def transform(p,q):
        transforms.append((list(p),list(q)))
        return (np.asarray(p)+[0,0,.0095]).tolist(),list(q)
    m=GroundedMonitor('G1','lift',lambda:state,motion_transform=transform)
    m.geometry_fn=lambda obs:({},0,np.empty((0,3)))
    m.begin_program('registered motion',0)
    m._phase_guard('approach',True,'current_phase')
    m._tracked(lambda name,args,kwargs:calls.append((name,args,kwargs)))('goto_pose',[[.5,0,.2],[0,0,1,0]],{'z_approach':.1})
    assert transforms==[([.5,0,.2],[0,0,1,0])]
    assert calls[0][1][0]==[.5,0,.20950000000000002]
    node=m.graph.nodes[m.last_call]
    assert node['data']['args'][0]==[.5,0,.2]
    m._tracked(lambda name,args,kwargs:calls.append((name,args,kwargs)))('open_gripper',[],{})
    assert len(transforms)==1
