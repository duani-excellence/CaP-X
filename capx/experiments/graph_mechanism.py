"""Offline dependency-control acceptance with real Sandbox actors and mock public APIs.

No simulator, vision service, LLM, or physical reward is used. Scripted encoder and
surface outputs exercise the production monitor and parent broker deterministically.
The control snippets here are mechanism fixtures, not model-generated demonstrations.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from capx.execution.graph_feedback import GRAPH_APIS, GraphFeedbackMonitor
from capx.execution.sandbox import SandboxProcess
from capx.experiments.pilot import dump

MOTIONS = {'goto_pose', 'open_gripper', 'close_gripper', 'home_pose'}


class ScriptedPublicAPI:
    """A small public-observation fixture, explicitly not a physics simulator."""
    def __init__(self, scenario):
        self.scenario = scenario
        self.state = dict(robot_cartesian_pos=np.array([.45, 0, .08, 1, 0, 0, 0, 1.]),
                          robot_joint_pos=np.zeros(7))
        self.positions = {'red cube': np.array([.45, 0., 0.]), 'green cube': np.array([.5, 0., 0.])}
        self.sizes = {'red cube': np.array([.04, .04, .04]), 'green cube': np.array([.05, .05, .05])}
        self.held = False
        self.support_reads = 0
        self.calls = []
        self.observations = []

    def observe(self):
        observation = {key: value.copy() for key, value in self.state.items()}
        self.observations.append(dict(index=len(self.observations),
            source='scripted_public_encoders_and_surface_estimates',
            robot={key: value.tolist() for key, value in observation.items()}, surfaces=self.evidence()))
        return observation

    def evidence(self):
        result = {}
        for name, position in self.positions.items():
            half = self.sizes[name] / 2
            result[name.split()[0]] = dict(center=position.tolist(), lo=(position-half).tolist(),
                                           hi=(position+half).tolist(), pixels=100, top_pixels=40)
        return result

    def call(self, name, args, kwargs):
        self.calls.append(dict(sequence=len(self.calls)+1, name=name, args=copy.deepcopy(args),
                               kwargs=copy.deepcopy(kwargs), before=self.observe()))
        if name == 'get_object_pose':
            object_name = args[0] if args else kwargs['object_name']
            if object_name == 'green cube':
                self.support_reads += 1
                if self.scenario == 'support_refresh' and self.support_reads == 2:
                    # Controlled new public estimate: no simulated object or hidden truth exists.
                    self.positions[object_name][0] += .004
            response = (self.positions[object_name].copy(), np.array([1., 0, 0, 0]), self.sizes[object_name].copy())
        elif name == 'sample_grasp_pose':
            response = (self.positions['red cube'] + np.array([0., 0., .03]), np.array([1., 0, 0, 0]))
        elif name == 'goto_pose':
            position = np.asarray(args[0] if args else kwargs['position'], float)
            quat = np.asarray(args[1] if len(args) > 1 else kwargs['quaternion_wxyz'], float)
            previous = self.state['robot_cartesian_pos'][:3].copy()
            self.state['robot_cartesian_pos'][:7] = np.r_[position, quat]
            if self.held:
                self.positions['red cube'] += position - previous
            response = None
        elif name == 'open_gripper':
            self.held = False
            self.state['robot_cartesian_pos'][7] = 1.
            response = None
        elif name == 'close_gripper':
            self.held = True
            self.state['robot_cartesian_pos'][7] = .5
            response = None
        else:
            raise ValueError('Unsupported mock API: ' + name)
        self.calls[-1]['after'] = self.observe()
        self.calls[-1]['response'] = copy.deepcopy(response)
        return response


def initial_code(scenario):
    # Targets intentionally belong only to this documented synthetic fixture.
    x_offset, height = (0., .081) if scenario == 'support_refresh' else (.02, .095)
    return f'''import numpy as np
open_gripper()
red = observe_object("red cube")
green = observe_object("green cube")
grasp = observe_grasp("red cube")
approach = plan_motion("approach", grasp["position"], grasp["quaternion"], [grasp["id"]])
run_plan(approach)
close_gripper()
robot = read_robot()
lift = plan_motion("lift", np.array(robot["tcp"][:3]) + np.array([0., 0., .12]), robot["tcp"][3:7], [robot["id"]])
run_plan(lift)
place_target = [green["position"][0] + {x_offset}, green["position"][1], {height}]
place = plan_motion("place", place_target, [1., 0., 0., 0.], [red["id"], green["id"]])
retreat = plan_motion("retreat", np.array(place_target) + np.array([0., 0., .1]), [1., 0., 0., 0.], [place])
run_plan(place)
open_gripper()
run_plan(retreat)
'''


def recovery_code(monitor, scenario):
    if scenario == 'local_alignment':
        return '''import numpy as np
robot = read_robot()
new_retreat = plan_motion("retreat", np.array(robot["tcp"][:3]) + np.array([0., 0., .1]), robot["tcp"][3:7], [robot["id"]])
run_plan(new_retreat)
'''
    lines = ['import numpy as np']
    for variable, key in [('red', 'object:red cube'), ('green', 'object:green cube')]:
        node = monitor.graph.latest.get(key)
        if node and monitor.graph.nodes[node]['valid']:
            lines.append(f'{variable} = recall_result({node!r})')
        else:
            lines.append(f'{variable} = observe_object({key.split(":", 1)[1]!r})')
    lines.extend([
        'place_target = [green["position"][0], green["position"][1], .081]',
        'place = plan_motion("place", place_target, [1., 0., 0., 0.], [red["id"], green["id"]])',
        'retreat = plan_motion("retreat", np.array(place_target) + np.array([0., 0., .1]), [1., 0., 0., 0.], [place])',
        'run_plan(place)', 'open_gripper()', 'run_plan(retreat)',
    ])
    return '\n'.join(lines) + '\n'


def run_case(root, mode, scenario):
    if mode not in ('G0', 'G1') or scenario not in ('support_refresh', 'local_alignment'):
        raise ValueError('Unknown mechanism case')
    dest = Path(root) / mode / scenario
    dest.mkdir(parents=True, exist_ok=False)
    fixture = ScriptedPublicAPI(scenario)
    broker_calls, pids, graph_events = [], [], []
    worker = None
    monitor = GraphFeedbackMonitor(mode, fixture.observe, fault='support_refresh' if scenario == 'support_refresh' else 'none',
                                   graph_emit=lambda event: graph_events.append(copy.deepcopy(event)),
                                   site=lambda: worker.call_site)
    # Explicitly offline scripted evidence, not claimed to be RGB-D or online Vision.
    monitor.evidence_fn = lambda public: fixture.evidence()

    def callback(name, args, kwargs):
        if worker.pid not in pids:
            pids.append(worker.pid)
        broker_calls.append(dict(block=monitor.block, actor_pid=worker.pid,
                                 name=name, site=copy.deepcopy(worker.call_site)))
        return monitor.invoke(name, args, kwargs, fixture.call)

    worker = SandboxProcess({name: None for name in GRAPH_APIS + ['open_gripper', 'close_gripper', 'home_pose', 'goto_pose']}, callback=callback)
    started = time.monotonic()
    result = dict(mode=mode, scenario=scenario, execution_mode='offline_mock_public_api_real_sandbox',
                  online_vision=False, llm_requests=0, physical_reward=None,
                  program_source='deterministic_mechanism_fixture_not_DeepSeek',
                  limitations=['No physical task success assertion', 'No real perception or motion',
                               'Intervention only at API boundaries'])
    try:
        code = initial_code(scenario)
        dump(dest/'block-0.json', dict(code=code, source=result['program_source']))
        monitor.begin_program(code, 0)
        result['initial'] = worker.run(code, timeout=10)
        dump(dest/'graph-after-block-0.json', monitor.graph.snapshot())
        place = next(node['id'] for node in monitor.graph.nodes.values() if node['kind'] == 'plan' and node['data']['skill'] == 'place')
        retreat = next(node['id'] for node in monitor.graph.nodes.values() if node['kind'] == 'plan' and node['data']['skill'] == 'retreat')
        lift = next(node['id'] for node in monitor.graph.nodes.values() if node['kind'] == 'plan' and node['data']['skill'] == 'lift')
        unrelated = monitor.graph.latest['grasp']
        initial_motion = [call['name'] for call in fixture.calls if call['name'] in MOTIONS]
        expected_initial = (['open_gripper', 'goto_pose', 'close_gripper', 'goto_pose'] if scenario == 'support_refresh'
                            else ['open_gripper', 'goto_pose', 'close_gripper', 'goto_pose', 'goto_pose', 'goto_pose', 'open_gripper'])
        result.update(old_place_invalid=not monitor.graph.nodes[place]['valid'],
                      old_retreat_invalid=not monitor.graph.nodes[retreat]['valid'],
                      unrelated_result_retained=monitor.graph.nodes[unrelated]['valid'],
                      blocked_action_not_dispatched=initial_motion == expected_initial,
                      node_ids=dict(old_place=place, old_retreat=retreat, completed_lift=lift, unrelated_grasp=unrelated),
                      motion_calls_before_recovery=initial_motion)
        dump(dest/'recovery-scope.json', monitor.summary())
        before_recovery = len(fixture.calls)
        code = recovery_code(monitor, scenario)
        dump(dest/'block-1.json', dict(code=code, source='deterministic_scope_fixture_not_DeepSeek'))
        worker.close()
        monitor.begin_program(code, 1)
        result['recovery'] = worker.run(code, timeout=10)
        recovery_motion = [call['name'] for call in fixture.calls[before_recovery:] if call['name'] in MOTIONS]
        result['motion_calls_during_recovery'] = recovery_motion
        result['new_retreat_verified'] = monitor.retreat_completed
        expected_recovery = ['goto_pose', 'open_gripper', 'goto_pose'] if scenario == 'support_refresh' else ['goto_pose']
        # Deliberately try to replay an already completed plan in a third fresh actor.
        before_replay = len(fixture.calls)
        code = f'run_plan({lift!r})'
        dump(dest/'block-2.json', dict(code=code, source='intentional_completed_action_replay_rejection'))
        worker.close()
        monitor.begin_program(code, 2)
        result['replay_attempt'] = worker.run(code, timeout=10)
        result['no_completed_motion_replayed'] = recovery_motion == expected_recovery and len(fixture.calls) == before_replay
        result['passed'] = all([
            result['initial']['status'] == 'interrupted', result['recovery']['status'] == 'ok',
            result['replay_attempt']['status'] == 'interrupted', result['old_place_invalid'],
            result['old_retreat_invalid'], result['blocked_action_not_dispatched'],
            result['no_completed_motion_replayed'], result['new_retreat_verified'],
            result['unrelated_result_retained'] == (mode == 'G1'), len(set(pids)) == 3,
        ])
    except Exception as exc:
        result.update(passed=False, error=type(exc).__name__ + ': ' + str(exc))
    finally:
        worker.close()
        result.update(seconds=time.monotonic()-started, actor_pids=pids,
                      broker_calls=broker_calls, raw_api_calls=len(fixture.calls),
                      local_recoveries=monitor.recoveries, fault_fired=monitor.fault_fired)
        dump(dest/'result.json', result)
        dump(dest/'graph-final.json', monitor.graph.snapshot())
        dump(dest/'public-observations.json', fixture.observations)
        dump(dest/'mock-api-calls.json', fixture.calls)
        dump(dest/'feedback-events.json', monitor.events)
        with (dest/'graph-events.jsonl').open('x') as stream:
            for event in graph_events:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', choices=['G0', 'G1'], default=['G0', 'G1'])
    parser.add_argument('--scenarios', nargs='+', choices=['support_refresh', 'local_alignment'],
                        default=['support_refresh', 'local_alignment'])
    args = parser.parse_args(argv)
    if len(set(args.groups)) != len(args.groups) or len(set(args.scenarios)) != len(args.scenarios):
        parser.error('Duplicate groups/scenarios')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = [run_case(root, mode, scenario) for scenario in args.scenarios for mode in args.groups]
    summary = dict(online_vision=False, llm_requests=0, physical_reward=None,
                   cases=len(rows), passed=sum(row['passed'] for row in rows), results=rows)
    dump(root/'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ['online_vision', 'llm_requests', 'physical_reward', 'cases', 'passed']}))
    if not all(row['passed'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
