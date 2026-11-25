"""Zero LLM / zero physical calls: re-search a historical public placement failure."""
import argparse,json,collections
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from capx.execution.bounded_ik import BoundedIK,IKSolveBudget
from capx.execution.control_frames import transform_target
from capx.execution.vision_skills import pose_matrix
from capx.execution.preplanning import DiagnosticReachability,search_placements
from capx.integrations.motion.pyroki import init_pyroki
from capx.execution.agent_events import plain


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True);p.add_argument('--object',required=True);p.add_argument('--target',required=True);p.add_argument('--ik-runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(x) for x in (a.episode/'trace.jsonl').read_text().splitlines()]
 idx=next(i for i,r in enumerate(rows) if r.get('kind')=='placement_quality_candidates' and r.get('object')==a.object)
 grasp_idx=max(i for i,r in enumerate(rows[:idx]) if r.get('kind')=='skill_grasp_plan' and r.get('object')==a.object)
 close=next(i for i in range(grasp_idx,idx) if rows[i].get('kind')=='api' and rows[i].get('name')=='close_gripper')
 cap=next(r['capture'] for r in rows[close+1:idx] if r.get('kind')=='public_capture')
 events=[json.loads(x) for x in (a.episode/'harness/events.jsonl').read_text().splitlines()]
 robot=next(r['data']['robot'] for r in events if r['type']=='ObservationCaptured' and r['data'].get('capture')==cap)
 h=dict(object=a.object,tcp=pose_matrix(robot['robot_cartesian_pos']),shape=rows[grasp_idx]['shape'])
 held=next(r for r in reversed(rows[:idx]) if r.get('kind')=='skill_hold_check' and r.get('object')==a.object)
 destination=next(r['fit'] for r in reversed(rows[:idx]) if r.get('kind')=='region_geometry_diagnostic' and r.get('role')=='destination')
 reference=np.array(next(r['joint_solution'] for r in reversed(rows[:idx]) if r.get('kind')=='ik' and r['status']=='PASS'))
 cal=json.loads((a.episode/'calibration.json').read_text());urdf=next(x['path'] for x in cal['sources'] if x['role']=='ik_urdf')
 def transform(p,q):
  pos,quat=transform_target(cal,p,q)
  return np.r_[quat,pos+Rotation.from_quat(np.roll(quat,-1)).apply(cal['api_tcp_to_hand_offset_m'])]
 solver=BoundedIK(init_pyroki(json.loads(a.ik_runtime.read_text())['url']),urdf_path=urdf,budget=IKSolveBudget(512))
 ik=DiagnosticReachability(solver,transform,lambda:reference,calibration=cal['calibration_hash'])
 s=SimpleNamespace(held=h,profile=json.loads((a.episode/'error-profile.json').read_text()),scene=SimpleNamespace(source=held['shape']['source']),reachable=ik)
 result=search_placements(s,dict(object=a.object,target=a.target),held['shape'],held['registration'],destination)
 repeat=search_placements(s,dict(object=a.object,target=a.target),held['shape'],held['registration'],destination)
 historical=[r for r in rows[idx:] if r.get('kind')=='ik'];last={r['call_generation']:r for r in historical}
 report=dict(mode='frozen_public_geometry_live_IK_NO_LLM_NO_MOTION',object=a.object,reference=str(a.episode),
  attachment=plain(h),geometry=destination,registration=held['registration'],result=result,
  historical=dict(iterations=len(historical),searches=len(last),unique_targets=len({tuple(r['target_wxyz_xyz']) for r in last.values()})),
  solver_calls=solver.budget.calls,exact_cache_hits=ik.hits,repeat_status=repeat['status'],physical_success_tested=False)
 (a.output/'diagnosis.json').write_text(json.dumps(plain(report),indent=2));(a.output/'ik-results.json').write_text(json.dumps(plain(ik.records),indent=2))
 print(json.dumps(dict(status=result['status'],tested=len(result['tested']),selected=result['selected'],solver_calls=solver.budget.calls,cache_hits=ik.hits)))
if __name__=='__main__':main()
