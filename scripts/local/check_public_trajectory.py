"""Frozen RGB-D check of a candidate joint trajectory, including held volume.

No simulator, LLM, controller, or motion dispatch. PASS covers sampled public
geometry only, not continuous collision certification or a release permit.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scripts.local.diagnose_region_memory import load_frame
from capx.execution.robot_self_geometry import PandaSelfGeometry
from capx.execution.open_motion import observed_free,tool_volume


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--pose-result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-samples',type=int,default=256);a=p.parse_args()
 if a.max_samples<1 or a.max_samples>1024:raise ValueError('Finite sample budget required')
 if a.output.exists():raise FileExistsError(a.output)
 d=json.loads(a.diagnosis.read_text());ep=Path(d['reference']);cal=json.loads((ep/'calibration.json').read_text());urdf=next(x['path'] for x in cal['sources'] if x['role']=='ik_urdf')
 plan=json.loads(a.pose_result.read_text())['mature_planner'];index=int(Path(plan['source']).name.split('-')[0]);frame,source=load_frame(ep,index)
 geometry=PandaSelfGeometry(urdf,cal);h=d['attachment'];h['tcp']=np.array(h['tcp'])
 s=SimpleNamespace(held=h,scene=SimpleNamespace(obs=frame),profile=json.loads((ep/'error-profile.json').read_text()))
 path=np.array(plan['trajectory']);fingers=np.abs(frame['robot_gripper_joint_pos'])
 mesh,_=geometry.mesh(frame['robot_joint_pos'][:7],finger_positions_m=fingers,observed_tcp=frame['robot_cartesian_pos'])
 tree=cKDTree(np.r_[mesh.vertices,mesh.triangles_center,tool_volume(s,frame['robot_cartesian_pos'][:7])]);checks=[];status='UNKNOWN';reason='sample_budget'
 if np.max(abs(path[0,:7]-frame['robot_joint_pos'][:7]))>.002:raise ValueError('Trajectory start is not the measured encoder state')
 for segment,(start,end) in enumerate(zip(path[:-1],path[1:])):
  count=max(1,int(np.ceil(np.max(abs(end[:7]-start[:7]))/.01)))
  for alpha in np.linspace(0,1,count+1)[1:]:
   if len(checks)>=a.max_samples:break
   mesh,meta=geometry.mesh((start*(1-alpha)+end*alpha)[:7],finger_positions_m=fingers)
   tcp=np.array(meta['public_fk_tcp']);q=np.roll(Rotation.from_matrix(tcp[:3,:3]).as_quat(),1)
   points=np.r_[mesh.vertices,mesh.triangles_center,tool_volume(s,np.r_[tcp[:3,3],q])];distance,_=tree.query(points)
   check=observed_free(points[distance>.004],frame);checks.append(dict(segment=segment,alpha=float(alpha),**check))
   if check['status']!='PASS':status='UNKNOWN';reason=check['reason'];break
  if checks[-1]['status']!='PASS' or len(checks)>=a.max_samples:break
 else:status='PASS_SAMPLED_ONLY';reason='all_sampled_new_volume_has_public_depth_evidence'
 result=dict(mode='frozen_public_path_diagnostic_NOT_execution',status=status,reason=reason,permission=False,source=source,attachment_included=True,checks=checks,physical_calls=0,new_llm_requests=0)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=status,last=checks[-1])))
if __name__=='__main__':main()
