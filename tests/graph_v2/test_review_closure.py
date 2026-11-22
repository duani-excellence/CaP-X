"""Actual isolated actors over public mock physics: dependency closure, not task success."""
import copy
import numpy as np
import pytest
from capx.execution.grounded_feedback import APIS
from capx.execution.sandbox import SandboxProcess
from test_release_permission_edges import release_ready


@pytest.mark.parametrize('mode',['G0','G1'])
def test_local_alignment_invalidates_old_retreat_then_new_actor_continues_without_replay(mode):
    m,state,surfaces,calls,raw,_=release_ready(mode)
    def held():
        m.record('L0','attachment_consistency','PASS','current_public_fixture',
            graph_parents=[m.attachment_id,m.geometry_ids['red cube']])
        s=copy.deepcopy(surfaces['red'])
        return {**s,'frame':'robot_base','height_m':.04,'bottom_z':s['lo'][2],
                'orientation_verified':True,'attachment_check_id':m.last_check}
    m._held_surface=held
    actors=[];receipts=[];release_checks=[]
    def callback(name,args,kwargs):
        if name=='open_gripper' and old_retreat:
            release_checks.append((not m.graph.nodes[old_retreat]['valid'],
                                   not m.graph.nodes[old_context]['valid']))
        return raw(name,args,kwargs)
    def run(code,block):
        m.begin_program(code,block)
        actor=SandboxProcess(dict.fromkeys(APIS+['open_gripper']),
            callback=lambda name,args,kwargs:m.invoke(name,args,kwargs,callback))
        try:
            result=actor.run(code,timeout=10)
            actors.append(actor.pid)
            receipts.append(result)
            return result
        finally:actor.close()
    old_retreat=old_context=None
    initial='''held=observe_surfaces('red cube')
support=observe_surfaces('green cube')
robot=read_robot()
place=plan_skill('place',{'robot':robot['id'],'held_surface':held['id'],'support_surface':support['id']})
run_plan(place)
robot=read_robot()
retreat=plan_skill('retreat',{'robot':robot['id']})
'''
    assert run(initial,0)['status']=='ok'
    place=next(n['id'] for n in m.graph.nodes.values() if n['kind']=='plan' and n['data']['skill']=='place')
    old_retreat=next(n['id'] for n in m.graph.nodes.values() if n['kind']=='plan' and n['data']['skill']=='retreat')
    old_context=m.placement
    assert old_context in m.graph.ancestors([place]) and old_context in m.graph.ancestors([old_retreat])
    assert m.graph.nodes[old_retreat]['valid']
    done_before=list(m.call_history);physical_before=len(calls)
    # A public fixture pose change requires release alignment. No private simulator data.
    state['robot_cartesian_pos'][0]+=.02
    for key in ('center','lo','hi'):surfaces['red'][key][0]+=.02
    for bound in surfaces['red']['top_xy_bounds']:bound[0]+=.02
    assert run("open_gripper()\nrun_plan('"+old_retreat+"')\n",1)['status']=='interrupted'
    assert release_checks==[(True,True)]  # Already invalid before release dispatch.
    assert not m.graph.nodes[old_retreat]['valid'] and m.phase=='released'
    assert m.local_attempts==1
    assert any(e['action']=='invalidate' and e['reason']=='local_alignment_changed_placement'
               and old_context in e['roots'] for e in m.graph.events)
    assert [name for name,_ in calls[physical_before:]]==['goto_pose','open_gripper']
    completed_after_release=list(m.call_history)
    assert run("robot=read_robot()\nretreat=plan_skill('retreat',{'robot':robot['id']})\nrun_plan(retreat)\n",2)['status']=='ok'
    assert m.phase=='ready' and m.retreat_completed and m.local_attempts==1
    assert len(m.call_history)==len(completed_after_release)+1
    assert m.call_history[:len(done_before)]==done_before
    assert all(m.graph.nodes[i]['state']=='done' for i in completed_after_release)
    assert all(sum(e.get('action')=='returned_call' and e.get('node_id')==i for e in m.graph.events)==1
               for i in completed_after_release)
    assert len(set(actors))==3
