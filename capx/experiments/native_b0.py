"""Minimal native task adapter for the existing CaP-X program-feedback runner.

Not an official paper reproduction and not the common Contract Skill baseline.
No Graph Executive, deterministic recovery, or Persistent Harness is installed.
The P0 sandbox, public sensors, bounded IK and independent score are retained.
"""
import copy
from types import SimpleNamespace
from pathlib import Path

TASK_DESCRIPTION = '''Place all four objects into their assigned quadrants of the
destination wooden bin (the other bin from the initial object bin). The assignment,
in robot-base X/Y, is: Milk low-X/low-Y; Bread high-X/low-Y;
Cereal low-X/high-Y; Can high-X/high-Y. Infer bin geometry from public Vision;
these are task instructions, not measured coordinates. Release the objects in
their assigned regions and retreat. No hidden object or bin coordinates are available.'''


class NativeProgramEnvironment:
    def __init__(self,root,seed,snapshot,api_budget):
        from .native_tasks import NativeBoundary
        from capx.execution.sandbox import SandboxProcess
        from capx.utils.request_audit import RequestAudit
        if snapshot is None or snapshot.get('seed')!=seed:
            raise ValueError('B0 native adapter requires the exact requested Snapshot')
        spec=copy.deepcopy(snapshot['spec'])
        if spec.get('api_budget',256)!=api_budget:
            raise ValueError('B0 API budget must match the frozen native scene')
        self.boundary=NativeBoundary(spec,seed,Path(root),mode='vision',expected_snapshot=snapshot)
        b=self.boundary;b.actor.close()
        # Explicitly separate original raw APIs from the default perception-only
        # NativeBoundary. G0/G1 actor allowlists remain unchanged.
        b.functions=b.api.functions();b.functions['goto_pose']=b.goto_pose
        self._functions=b.functions;self._apis={'FrankaControlApi':b.api}
        self._actor=b.actor=SandboxProcess(b.functions,callback=b.call)
        self.low_level_env=b.low;self.cfg=SimpleNamespace(execution_timeout=240)
        self._trace_path=Path(root)/'trace.jsonl'
        self._request_audit=RequestAudit(Path(root)/'requests.jsonl').__enter__()
        self.reset_info=dict(initial_state_hash=b.initial['state_hash'],snapshot_replay_verified=True)
        self._task_prompt=TASK_DESCRIPTION
        self._full_prompt=[{'role':'system','content':'Generate a complete Python robot program using only the listed APIs and numpy. No files, network, env, obs, globals or private attributes. Physical success is independently evaluated.'},
            {'role':'user','content':self._task_prompt+'\nAPI documentation:\n'+b.api.combined_doc()}]

    @property
    def _api_calls(self):return self.boundary.calls

    def _get_observation(self):return self.boundary.capture()

    def _write(self,kind,**data):self.boundary.log(kind,**data)

    def evaluation_state(self):
        from .native_tasks import physical_state
        return physical_state(self.low_level_env.robosuite_env)

    def step(self,code):
        from .native_tasks import physical_reward
        b=self.boundary;before=b.motion
        result=self._actor.run(code,timeout=self.cfg.execution_timeout)
        if result['ok'] and b.motion>before:
            for _ in range(15):b.low._step_once()
        private=self.evaluation_state()
        truncated=b.low._sim_step_count>=b.low.max_steps or result['status'] in ('timeout','system_error')
        reward=None if result['status'] in ('timeout','system_error') else physical_reward(
            result['ok'] and not truncated,b.motion-before,private['raw_success'])
        info=dict(execution_status=result['status'],stdout=result.get('stdout',''),stderr=result.get('stderr',''),
            physical_success=private['raw_success'],api_calls=b.calls)
        self._write('program_result',**info,reward=reward,truncated=truncated)
        return self._get_observation(),reward,reward==1,truncated,info

    def enable_video_capture(self,*args,**kw):return self.low_level_env.enable_video_capture(*args,**kw)
    def get_video_frames(self,**kw):return self.low_level_env.get_video_frames(**kw)
    def close(self):
        self._request_audit.__exit__()
        self.boundary.close()


def make_environment(task,root,seed,snapshot,api_budget):
    if task!='pick_place':raise ValueError('Native B0 task adapter not available')
    return NativeProgramEnvironment(root,seed,snapshot,api_budget)


def run_episode(output,snapshot,config,budget,seed,repeat,source_hashes):
    from .graph_v2 import run_episode as original_runner
    cfg=copy.deepcopy(config)
    for key in ('api_budget','episode_seconds'):
        cfg['budgets'][key]=snapshot['spec'][key]
    result,_=original_runner(output,'pick_place',seed,repeat,snapshot,cfg,budget,'B0',
        source_hashes=source_hashes,environment_factory=make_environment,condition_text=TASK_DESCRIPTION)
    result.update(method='CaP-X-program-feedback-minimal-task-adapter',official_original_reproduction=False,
        strategy='complete Python program; original console-feedback replanning',
        common_contract_skills=False,persistent_harness=False,graph_executive=False,
        raw_success=result.get('final_physics',{}).get('raw_success',False),
        native_progress=result.get('final_physics',{}).get('native_progress',{}))
    # Native result provenance is an additional record; original runner evidence is immutable.
    from .graph_v2 import save
    save(Path(output)/'episodes/B0/adapter-provenance.json',result)
    return result
