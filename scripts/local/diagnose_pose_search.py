"""Bounded frozen-public pose search and mature-planner probe; no LLM or motion.

Consumes diagnose_preplanning.py output. Tilt candidates are diagnostic only;
they cannot authorize the existing upright release capability.
"""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from capx.execution.preplanning import placement_candidates,DiagnosticReachability
from capx.execution.bounded_ik import BoundedIK,IKSolveBudget
from capx.execution.control_frames import transform_target
from capx.integrations.motion.pyroki import init_pyroki
from capx.execution.agent_events import plain


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--ik-runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--tilts',type=float,nargs='+',default=[0,20,40]);p.add_argument('--yaw-step',type=int,default=45);p.add_argument('--max-solver-calls',type=int,default=512);p.add_argument('--mature',action='store_true')
 a=p.parse_args()
 if not 15<=a.yaw_step<=180 or not 8<=a.max_solver_calls<=2400 or any(not 0<=t<=45 for t in a.tilts):raise ValueError('Invalid bounded diagnostic search')
 a.output.mkdir(parents=True,exist_ok=False);d=json.loads(a.diagnosis.read_text());ep=Path(d['reference']);cal=json.loads((ep/'calibration.json').read_text());urdf=next(x['path'] for x in cal['sources'] if x['role']=='ik_urdf')
 ikrows=json.loads(a.diagnosis.with_name('ik-results.json').read_text());reference=np.array(next(iter(ikrows.values()))['reference'])
 def transform(p,q):
  pos,quat=transform_target(cal,p,q);return np.r_[quat,pos+Rotation.from_quat(np.roll(quat,-1)).apply(cal['api_tcp_to_hand_offset_m'])]
 solver=BoundedIK(init_pyroki(json.loads(a.ik_runtime.read_text())['url']),urdf_path=urdf,budget=IKSolveBudget(a.max_solver_calls))
 ik=DiagnosticReachability(solver,transform,lambda:reference,calibration=cal['calibration_hash'])
 base=Rotation.from_quat([0,1,0,0]);orientations=[np.roll((Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_euler('y',tilt,degrees=True)*base).as_quat(),1) for tilt in a.tilts for yaw in range(0,360,a.yaw_step)]
 h=d['attachment'];h['tcp']=np.array(h['tcp']);profile=json.loads((ep/'error-profile.json').read_text())
 candidates=placement_candidates(h,h['shape'],d['registration'],d['result']['region'],profile,diagnostic_orientations=orientations,max_positions=25)
 rows=[];solution=None
 for r in candidates:
  if solver.budget.calls+16>a.max_solver_calls:break
  check=ik.query(r['position'],r['quaternion_wxyz']);rows.append(dict(candidate=r,ik=check['id'],status=check['status']))
  if check['status']=='KINEMATIC_SOLUTION':
   above=ik.query(np.array(r['position'])+[0,0,.2],r['quaternion_wxyz'])
   if above['status']=='KINEMATIC_SOLUTION':solution=dict(candidate=r,release=check,above=above);break
 result=dict(mode='frozen_public_pose_diagnostic_NOT_execution',physical_calls=0,new_llm_requests=0,tested=rows,solution=solution,solver_calls=solver.budget.calls,permission=False)
 if a.mature and solution:
  from scripts.local.diagnose_region_memory import load_frame
  from capx.integrations.motion.public_planning import plan_pyroki
  from capx.execution.robot_self_geometry import PandaSelfGeometry
  from capx.execution.open_motion import PublicJointSweep
  from types import SimpleNamespace
  log=[json.loads(x) for x in (ep/'trace.jsonl').read_text().splitlines()]
  stop=next(i for i,r in enumerate(log) if r.get('kind')=='placement_quality_candidates' and r.get('object')==d['object'])
  index=next(r['capture'] for r in reversed(log[:stop]) if r.get('kind')=='public_capture')
  frame,source=load_frame(ep,index);start=np.r_[frame['robot_joint_pos'][:7],abs(frame['robot_gripper_joint_pos'][0])]
  began=time.monotonic()
  try:
   sphere=Path(__file__).resolve().parents[2]/'capx/serving/assets/panda_spheres.json'
   floor=d['geometry']['floor_z']+cal['T_ik_base_public_base'][2][3]
   plan=plan_pyroki(urdf,sphere,start,solution['above']['target_hand_wxyz_xyz'],[dict(type='halfspace',point=[0,0,floor],normal=[0,0,1])],held=True)
   geometry=PandaSelfGeometry(urdf,cal);events=[];checker=PublicJointSweep(geometry,SimpleNamespace(obs=frame,source=source),lambda k,**v:events.append(dict(kind=k,**v)))
   try:checker(np.array(plan['trajectory'][1]));first=dict(status='PASS_FIRST_SEGMENT_ONLY')
   except ValueError as exc:first=dict(status='REJECTED',reason=str(exc))
   plan.update(first_segment_check=dict(first,events=events),source=source,start_joints=start.tolist())
  except Exception as exc:plan=dict(status='UNAVAILABLE',error=str(exc),traceback=traceback.format_exc())
  plan['seconds']=time.monotonic()-began;result['mature_planner']=plan
 (a.output/'results.json').write_text(json.dumps(plain(result),indent=2));(a.output/'ik-results.json').write_text(json.dumps(plain(ik.records),indent=2))
 print(json.dumps(dict(tested=len(rows),solution=solution['candidate'] if solution else None,solver_calls=solver.budget.calls,mature_validation=result.get('mature_planner',{}).get('validation'))))
if __name__=='__main__':main()
