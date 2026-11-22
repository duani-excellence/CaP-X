"""Check public renderer/geometry before paying for a grounded initial program."""
import pytest
from capx.experiments import graph_v2 as runner
from capx.execution.public_camera import DepthCalibrationUnavailable
from test_runner_v2 import fake,Monitor,Env,Budget


@pytest.mark.parametrize('group',['G0','G1'])
@pytest.mark.parametrize('error',[DepthCalibrationUnavailable,ValueError])
def test_public_preflight_failure_never_queries_or_creates_initial(tmp_path,fake,monkeypatch,group,error):
    class Unavailable(Monitor):
        def capture(self):raise error('public calibration unavailable')
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:Unavailable())
    budget=Budget()
    row,initial=runner.run_episode(tmp_path,'stack',21,0,{'state_hash':'same'},fake,budget,group)
    assert budget.count==0 and initial is None
    assert row['status']=='system_error' and row['policy_success'] is False
    assert row['stop_reason']=='public_observation_preflight_failed'
    assert row['public_preflight']['error']==error.__name__
    assert row['actual_api_calls']==0 and row['executed'] is False and row['blocks']==[]
    assert row['new_llm_requests']==0 and row['llm_requests']==0
    assert Env.created[0].closed and not Env.created[0].steps
    assert not (tmp_path/'initial'/'G0-G1.json').exists()


def test_cached_initial_does_not_bypass_public_preflight(tmp_path,fake,monkeypatch):
    budget=Budget()
    _,initial=runner.run_episode(tmp_path/'first','stack',21,0,{'state_hash':'same'},fake,budget,'G0')
    class Unavailable(Monitor):
        def capture(self):raise DepthCalibrationUnavailable('unavailable')
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:Unavailable())
    row,returned=runner.run_episode(tmp_path/'cached','stack',21,0,{'state_hash':'same'},fake,budget,'G1',initial)
    assert budget.count==1 and returned is initial
    assert row['status']=='system_error' and not row['executed'] and row['actual_api_calls']==0
    assert row['new_llm_requests']==0 and not (tmp_path/'cached'/'initial').exists()


def test_failed_first_arm_leaves_second_arm_to_generate_real_shared_initial(tmp_path,fake,monkeypatch):
    class Unavailable(Monitor):
        def capture(self):raise DepthCalibrationUnavailable('unavailable')
    count=[]
    def install(*a,**kw):
        count.append(1);return Unavailable() if len(count)==1 else Monitor()
    monkeypatch.setattr(runner,'install_monitor',install)
    budget=Budget();result=runner.run_pair(tmp_path,'stack',21,0,{'state_hash':'same'},fake,budget,groups=('G0','G1'))
    first,second=result['episodes']
    assert budget.count==1
    assert first['status']=='system_error' and first['new_llm_requests']==0
    assert second['status']=='success' and second['initial_generated_here'] is True
    assert len(Env.created[0].steps)==0 and len(Env.created[1].steps)==1


def test_b0_does_not_receive_grounded_preflight(tmp_path,fake,monkeypatch):
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:pytest.fail('B0 has its own legal raw API'))
    budget=Budget();row,_=runner.run_episode(tmp_path,'stack',21,0,{'state_hash':'same'},fake,budget,'B0')
    assert row['status']=='success' and budget.count==1
    assert 'public_preflight' not in row


def test_successful_preflight_is_before_query_and_not_a_later_permission(tmp_path,fake,monkeypatch):
    events=[]
    class Fresh(Monitor):
        def capture(self):events.append('capture')
        def begin_program(self,code,block):super().begin_program(code,block);self.capture()
    class Ordered(Budget):
        def query(self,*a):events.append('query');return super().query(*a)
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:Fresh())
    budget=Ordered();row,_=runner.run_episode(tmp_path,'stack',21,0,{'state_hash':'same'},fake,budget,'G1')
    assert events==['capture','query','capture']
    assert row['public_preflight']['status']=='PASS' and row['public_preflight']['seconds']>=0
    assert row['status']=='success'


def test_preflight_time_uses_original_episode_deadline(tmp_path,fake,monkeypatch):
    from types import SimpleNamespace
    clock=[0.];deadlines=[]
    monkeypatch.setattr(runner,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    class Slow(Monitor):
        def capture(self):clock[0]+=37.
    class Timed(Budget):
        def query(self,*a):deadlines.append(self.deadline);return super().query(*a)
    monkeypatch.setattr(runner,'install_monitor',lambda *a,**kw:Slow())
    row,_=runner.run_episode(tmp_path,'stack',21,0,{'state_hash':'same'},fake,Timed(),'G0')
    assert deadlines==[fake['budgets']['episode_seconds']]
    assert row['public_preflight']['seconds']==37. and row['total_seconds']>=37.
