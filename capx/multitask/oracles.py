"""Narrow oracle compatibility fixes; no change to isolation policy or task reward."""
NUT = '''import numpy as np

def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)], [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)], [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

handle, quat = sample_grasp_pose("extruded handle of the brown square nut")
center, unused = get_object_pose("white hollow center of the brown square nut")
peg, peg_quat = get_object_pose("square block")
r = rotation(quat)
if np.dot(center-handle,r[:,0]) < 0:
    w,x,y,z=quat
    quat=np.array([-z,y,-x,w])
    r=rotation(quat)
offset = r.T @ (handle-center)
open_gripper()
goto_pose(handle,quat,z_approach=0.05)
close_gripper()
goto_home_joint_position()
target=peg+rotation(peg_quat)@offset
goto_pose(target,peg_quat,z_approach=0.05)
goto_pose(target+np.array([0,0,-0.02]),peg_quat)
open_gripper()
goto_home_joint_position()
'''
WIPE = '''import numpy as np
pos, unused, extent=get_object_pose("brown spill",return_bbox_extent=True)
xs=np.linspace(pos[0]-extent[0]/2,pos[0]+extent[0]/2,7)
ys=np.linspace(pos[1]-extent[1]/2,pos[1]+extent[1]/2,6)
q=np.array([0,0,1,0])
goto_pose(np.array([xs[0],ys[0],0.1]),q)
for i,y in enumerate(ys):
    path=xs if i%2==0 else xs[::-1]
    for x in path:
        goto_pose(np.array([x,y,0.0]),q)
goto_pose(np.array([path[-1],ys[-1],0.1]),q)
'''
