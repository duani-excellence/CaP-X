import numpy as np
import pytest
from capx.execution.grounded_feedback import GroundedMonitor
from capx.execution.sandbox import ExecutionInterrupted


def fixture(mode='G1',task='stack'):
 state={'robot_cartesian_pos':np.array([.5,0,.14,0,0,1,0,.5]),'robot_joint_pos':np.zeros(8),'robot_gripper_joint_pos':np.array([.04,-.04])}
 m=GroundedMonitor(mode,task,lambda:state)
 m.close_tcp_z=.02  # Explicit public reference for manually constructed raised-hold fixtures.
 surfaces={'red':{'center':[.5,0,.02],'lo':[.48,-.02,0],'hi':[.52,.02,.04]},'green':{'center':[.65,0,.025],'lo':[.625,-.025,0],'hi':[.675,.025,.05]}}
 for value in surfaces.values():
  value.update(top_xy_bounds=[value['lo'][:2].copy(),value['hi'][:2].copy()],top_pixels=400,pixels=600)
 m.geometry_fn=lambda obs:(surfaces,0,np.empty((0,3)))
 def visible_top_mesh(obs,name):
  surface=surfaces['red' if name=='red cube' else 'green']
  x=np.linspace(surface['lo'][0],surface['hi'][0],21);y=np.linspace(surface['lo'][1],surface['hi'][1],21)
  xx,yy=np.meshgrid(x,y)
  return np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,surface['hi'][2])]
 m.color_points_fn=visible_top_mesh
 calls=[]
 def raw(name,args,kw):
  calls.append((name,args))
  if name=='get_object_pose':return np.array([.5,0,.02]),np.array([0,0,1,0]),np.array([.04]*3)
  if name=='sample_grasp_pose':return np.array([.5,0,.02]),np.array([0,0,1,0])
  if name=='goto_pose':
   delta=np.asarray(args[0])-state['robot_cartesian_pos'][:3]
   state['robot_cartesian_pos'][:7]=np.r_[args[0],args[1]]
   if m.closed:
    s=surfaces['red' if m.held_object=='red cube' else 'green']
    for k in ('center','lo','hi'):s[k]=(np.asarray(s[k])+delta).tolist()
 m.begin_program('program',0)
 return m,state,surfaces,calls,raw


def test_duplicate_close_cannot_reset_holding_phase():
 m,s,v,c,f=fixture();m.phase='holding_verified';m.closed=True;m.held_object='red cube';m.lifted=True
 with pytest.raises(ExecutionInterrupted,match='already_holding'):m.invoke('close_gripper',[],{},f)
 assert not c and m.phase=='holding_verified' and m.lifted


def test_release_cannot_bypass_required_retreat():
 m,s,v,c,f=fixture();m.phase='released';m.released=True
 for name in ('open_gripper','close_gripper','home_pose'):
  with pytest.raises(ExecutionInterrupted,match='retreat'):m.invoke(name,[],{},f)
 assert not c


def test_unverified_hold_cannot_lift_repeatedly():
 m,s,v,c,f=fixture();m.phase='holding_unverified';m.closed=True;m.held_object='red cube';m.lift_attempts=1
 r=m.invoke('read_robot',[],{},f)
 p=m.invoke('plan_skill',['lift',{'robot':r['id']}],{},f)
 with pytest.raises(ExecutionInterrupted,match='verify_hold'):m.invoke('run_plan',[p],{},f)
 assert not c


def test_raw_motion_and_forged_source_data_are_rejected():
 m,s,v,c,f=fixture()
 with pytest.raises(ExecutionInterrupted):m.invoke('goto_pose',[[.1,0,.1],[0,0,1,0]],{},f)
 with pytest.raises(ExecutionInterrupted):m.invoke('plan_skill',['approach',{'grasp':{'id':'fake','position':[0,0,0]}}],{},f)
 assert not c


def test_hold_probe_can_recover_without_historical_baseline():
 m,s,v,c,f=fixture();m.phase='holding_unverified';m.closed=True;m.held_object='red cube';m.lift_attempts=1
 result=m.invoke('verify_hold',[],{},f)
 assert result['status']=='PASS' and m.phase=='holding_verified'
 assert [x[0] for x in c]==['goto_pose','goto_pose']
 assert all(abs(x[1][0][2]-.14)<1e-8 for x in c)


def test_motion_invalidates_held_pose_but_not_other_object():
 m,s,v,c,f=fixture()
 red=m.invoke('observe_object',['red cube'],{},f);green=m.invoke('observe_object',['green cube'],{},f)
 m.phase='holding_verified';m.closed=True;m.lifted=True;m.held_object='red cube';m.baseline={'top_z':-.1,'tcp_z':0}
 r=m.invoke('read_robot',[],{},f);p=m.invoke('plan_skill',['lift',{'robot':r['id']}],{},f)
 m.invoke('run_plan',[p],{},f)
 assert not m.graph.nodes[red['id']]['valid']
 assert m.graph.nodes[green['id']]['valid']


def test_new_actor_preserves_stage_and_recovery_budget():
 m,s,v,c,f=fixture();m.phase='holding_verified';m.closed=True;m.held_object='red cube';m.local_attempts=3;m.hold_probes=1
 m.begin_program('recovery',1)
 assert m.phase=='holding_verified' and m.local_attempts==3 and m.hold_probes==1


def test_role_binding_rejects_robot_as_grasp():
 m,s,v,c,f=fixture();r=m.invoke('read_robot',[],{},f)
 with pytest.raises(ExecutionInterrupted):m.invoke('plan_skill',['approach',{'grasp':r['id']}],{},f)
 assert not c


def test_release_invalidates_last_held_pose():
 m,s,v,c,f=fixture();r=m.invoke('observe_object',['red cube'],{},f)
 m.closed=True;m.held_object='red cube';m.phase='holding_verified'
 m._effects('open_gripper')
 assert not m.graph.nodes[r['id']]['valid']


def test_public_grasp_attachment_survives_new_actor_and_replaces_occluded_centroid():
 m,s,v,c,f=fixture();m.invoke('observe_object',['red cube'],{},f)
 g=m.invoke('observe_grasp',['red cube'],{},f);p=m.invoke('plan_skill',['approach',{'grasp':g['id']}],{},f)
 m.invoke('run_plan',[p],{},f);m.invoke('close_gripper',[],{},f)
 assert m.attachment and m.attachment_id
 r=m.invoke('read_robot',[],{},f);p=m.invoke('plan_skill',['lift',{'robot':r['id']}],{},f);m.invoke('run_plan',[p],{},f)
 expected=np.array(v['red']['center']);v['red']['center'][0]+=.008
 m.begin_program('new actor',1)
 observed=m.invoke('observe_surfaces',['red cube'],{},f)
 np.testing.assert_allclose(observed['center'],expected)
 assert any(e['source']==m.attachment_id and e['target']==observed['id'] for e in m.graph.edges)
 assert observed['source']=='public_pregrasp_center_and_current_robot_rigid_attachment'


def test_orientation_permission_has_its_own_graph_source():
 m,s,v,c,f=fixture();m.invoke('observe_object',['red cube'],{},f);old=m.geometry_ids['red cube']
 g=m.invoke('observe_grasp',['red cube'],{},f);p=m.invoke('plan_skill',['approach',{'grasp':g['id']}],{},f);m.invoke('run_plan',[p],{},f)
 new=m.geometry_ids['red cube']
 assert new!=old and m.graph.nodes[new]['data']['orientation_verified']
 assert any(e['source']==old and e['target']==new for e in m.graph.edges)


def test_place_uses_precision_control_and_corrections_share_episode_budget(monkeypatch):
 import capx.execution.grounded_feedback as module
 m,s,v,c,f=fixture();m.closed=True;m.lifted=True;m.phase='holding_verified';m.held_object='red cube'
 m.geometry['red cube']={'height_m':.04,'orientation_verified':True}
 seen=[]
 def precise(p,q,**kw):
  seen.append(kw['lift_verified']);assert kw['reserve_correction']() is True
  kw['execute']('goto_pose',[p,q],{'z_approach':0.})
  return {'status':'PASS','corrections':1}
 monkeypatch.setattr(module,'precise_goto',precise)
 h=m.invoke('observe_surfaces',['red cube'],{},f);g=m.invoke('observe_surfaces',['green cube'],{},f);r=m.invoke('read_robot',[],{},f)
 plan=m.invoke('plan_skill',['place',{'held_surface':h['id'],'support_surface':g['id'],'robot':r['id']}],{},f)
 m.invoke('run_plan',[plan],{},f)
 assert seen==[True] and m.local_attempts==1 and m.recoveries==1
