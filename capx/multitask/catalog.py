"""One task catalog shared by CLI and Web. Availability requires real acceptance evidence."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
TASKS={
 'stack':dict(label='方块堆叠',module='franka_pick_place',low='robosuite_cubes.FrankaRobosuiteCubesLowLevel',api='control_privileged.FrankaControlPrivilegedApi',criterion='Released stable Stack success'),
 'lift':dict(label='方块抬升',module='franka_lift',low='robosuite_cube_lift.FrankaRobosuiteCubeLiftLowLevel',api='control_privileged.FrankaControlPrivilegedApi',criterion='Lift success AND bilateral grasp, held stable for 0.5 s'),
 'restack':dict(label='重新堆叠',module='franka_cube_restack',low='robosuite_cubes_restack.FrankaRobosuiteCubesRestackLowLevel',api='control_privileged.FrankaControlPrivilegedApi',criterion='Reverse initial stack; released stable Stack success'),
 'nut_assembly':dict(label='螺母装配',module='franka_nut_assembly',low='robosuite_nut_assembly.FrankaRobosuiteNutAssembly',api='nut_assembly_privileged.FrankaControlNutAssemblyPrivilegedApi',criterion='Nut on correct peg AND released/still for 0.5 s'),
 'wipe':dict(label='污渍擦拭',module='franka_spill_wipe',low='robosuite_spill_wipe.FrankaRobosuiteSpillWipeLowLevel',api='spill_wipe_privileged.FrankaControlSpillWipePrivilegedApi',criterion='All wipe markers cleared, not dense fraction reward'),
 'handover':dict(label='双臂交接',module='two_arm_handover',low='robosuite_handover.RobosuiteHandoverEnv',api='handover_privileged.FrankaHandoverPrivilegedApi',criterion='Receiver grasps hammer handle, giver released, height threshold, 0.5 s'),
 'two_arm_lift':dict(label='双臂抬举',module='two_arm_lift',low='robosuite_two_arm_lift.RobosuiteTwoArmLiftEnv',api='two_arm_lift_privileged.FrankaTwoArmLiftPrivilegedApi',criterion='Pot bottom > table +0.10 m AND both handle grasps for 0.5 s'),
}
def evidence():
 p=ROOT/'docs/multitask/acceptance.json'
 return json.loads(p.read_text()) if p.exists() else {}
def catalog():
 e=evidence()
 rows=[dict(task=k,**v,**e.get(k,{'status':'unverified','reason':'No real acceptance yet'})) for k,v in TASKS.items()]
 rows.extend([dict(task='libero_open_microwave',label='LIBERO 打开微波炉',status='blocked',reason='已有适配；当前 Robosuite 缺少 single_arm_env，需隔离依赖栈'),dict(task='libero_pick_alphabet_soup',label='LIBERO 抓取汤罐',status='blocked',reason='已有适配；与当前 Robosuite 不兼容，未实测验收'),dict(task='r1pro_behavior',label='R1Pro / BEHAVIOR',status='blocked',reason='OmniGibson / BDDL 未安装；本轮不新建重型平台')])
 return rows
def require_ready(task):
 if task not in TASKS:raise ValueError('Unknown task')
 row=next(x for x in catalog() if x['task']==task)
 if row['status']!='ready':raise ValueError(f"{task}: {row['status']}: {row.get('reason','not accepted')}")
