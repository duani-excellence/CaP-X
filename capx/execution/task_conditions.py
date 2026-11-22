"""Task predicates backed by the EXISTING parent-owned DependencyGraph.

No environment or evaluator is accepted here. Returned calls are history;
expected effects are obligations; only public checks establish current truth.
The small task recipes express ordering constraints, not graph-topological
robot motion. All physical dispatch still goes through GroundedMonitor.
"""
from __future__ import annotations

import copy
import numpy as np

from .public_geometry import place_metrics
from .support_geometry import reliable_top


STAGES = {
    'stack': (('red cube', 'green cube'),),
    'lift': (('red cube', 'held'),),
    'restack': (('green cube', 'table'), ('red cube', 'green cube')),
}

# Executable contracts are paired with the existing numerical/phase gates.
# A returned motion never, by itself, satisfies its listed physical effects.
CONTRACTS = {
    'approach': {'data': ['grasp'], 'requires': ['empty', 'task_stage_ready'],
                 'expected': ['approached'], 'verify': 'TCP convergence and open sweep'},
    'close_gripper': {'data': [], 'requires': ['approached'],
                      'expected': ['holding'], 'verify': 'lift or visual co-motion (not jaw obstruction)'},
    'lift': {'data': ['robot'], 'requires': ['closure_returned'],
             'expected': ['holding', 'lifted'], 'verify': 'public baseline rise or verify_hold'},
    'verify_hold': {'data': [], 'requires': ['closure_returned', 'lift_attempted'],
                    'expected': ['holding', 'lifted'], 'verify': 'bounded public co-motion'},
    'place': {'data': ['robot','held_surface','support_surface'], 'requires': ['holding'],
              'expected': ['aligned'], 'verify': 'current public pre-release geometry'},
    'table_place': {'data': ['robot','held_surface','table_region'], 'requires': ['holding'],
                    'expected': ['aligned'], 'verify': 'current table clearance and held geometry'},
    'open_gripper': {'data': [], 'requires': ['current_release_permission'],
                     'expected': ['placed','empty'], 'verify': 'post-release public geometry; return is not effect'},
    'retreat': {'data': ['robot'], 'requires': ['release_returned'],
                'expected': ['safe_retreat'], 'verify': 'current TCP convergence after release'},
}


class TaskConditions:
    def __init__(self, graph, task):
        if task not in STAGES: raise ValueError('Unsupported condition task')
        self.graph, self.task = graph, task
        self.stages = STAGES[task]
        self.ids = {}
        self.history = {}
        self.positioned_destinations = {}
        self.contract = graph.add('capability', {'type':'task_conditions_v22',
            'task':task, 'stages':self.stages, 'contracts':CONTRACTS,
            'reward':'independent_evaluator_only'})
        for key in self.goal_keys:
            self.set(key, 'UNKNOWN', reason='not_yet_observed')
        self.goal_node = graph.add('task_goal', {'task':task, 'all_of':self.goal_keys,
            'ordering':self.stages, 'not_a_reward':True}, parents=[self.contract])
        self.stage_goals={}
        previous=None;previous_key=None
        for obj,dest in self.stages:
            key='holding:'+obj if dest=='held' else 'placed:'+obj+':'+dest
            self.stage_goals[key]=graph.add('task_subgoal',{'condition':key,
                'object':obj,'destination':dest,'completion_requires_public_verification':True,
                'requires_conditions':[previous_key] if previous_key else [],
                'previous_stage':previous},parents=[self.goal_node]+([(previous,'precondition')] if previous else []))
            previous=self.stage_goals[key]
            previous_key=key

    @property
    def goal_keys(self):
        if self.task == 'lift': return ['holding:red cube', 'lifted:red cube']
        return ['placed:red cube:green cube', 'safe_retreat']

    def get(self, key):
        node = self.graph.nodes.get(self.ids.get(key))
        if node is None:
            return {'key':key, 'truth':'UNKNOWN', 'reason':'no_evidence', 'id':None}
        row = copy.deepcopy(node['data']); row['id'] = node['id']
        if not node['valid']:
            row.update(truth='UNKNOWN', reason='evidence_invalidated', invalidated_by=node['invalidated_by'][-3:])
        return row

    def set(self, key, truth, *, evidence=(), epochs=None, reason, dependencies=(), metrics=None):
        if truth not in ('TRUE','FALSE','UNKNOWN'): raise ValueError('Invalid condition truth')
        evidence = list(dict.fromkeys(evidence))
        self.graph._ids(evidence)
        deps = [self.ids[k] for k in dependencies if k in self.ids]
        if (any(not self.graph.nodes[n]['valid'] for n in evidence+deps)
                or len(deps) != len(dependencies)):
            truth = 'UNKNOWN'
        previous = self.ids.get(key)
        data = {'key':key, 'predicate':key.split(':')[0], 'roles':key.split(':')[1:],
                'truth':truth, 'epochs':epochs or {}, 'reason':reason,
                'evidence_ids':evidence, 'condition_dependencies':list(dependencies)}
        if metrics is not None:data['metrics']=copy.deepcopy(metrics)
        old = self.get(key)
        # A fresh attestation with the same state/epoch does not invalidate a
        # pending action. Pose handles have their own, stricter version rules.
        if previous and self.graph.nodes[previous]['valid'] and all(old.get(k)==data[k] for k in ('truth','epochs','reason','condition_dependencies')):
            self.graph._event('condition_attested', condition_id=previous, key=key,
                              evidence_ids=evidence, truth=truth, metrics=copy.deepcopy(metrics))
            return previous
        if previous:self.graph.invalidate([previous], 'condition_updated:'+key)
        node = self.graph.version('condition:'+key, 'condition', data,
            parents=[(n,'attestation' if self.graph.nodes[n]['kind']=='verification_fact' else 'data')
                     for n in evidence]+[(n,'precondition') for n in deps], invalidate=False)
        self.ids[key] = node
        self.graph._event('condition_transition', key=key, previous=previous, current=node,
                          before=old['truth'], after=truth, reason=reason)
        return node

    def expect(self, keys, call):
        return self.graph.add('expected_effect', {'conditions':list(keys),
            'verified':False, 'returned_is_not_effect':True}, parents=[call])

    def returned(self, skill, obj, dest, call, plan=None):
        # Task goals are not execution provenance. Bind actual placement to its
        # skill/source, then carry that destination through release and retreat.
        if skill=='table_place':dest='table'
        elif skill=='place' and plan:
            inputs=self.graph.nodes[plan]['data'].get('inputs',{})
            support=self.graph.nodes.get(inputs.get('support_surface'),{})
            dest=support.get('data',{}).get('object')
        elif skill in ('release','retreat'):
            dest=self.positioned_destinations.get(obj,dest)
        if skill in ('place','table_place'):self.positioned_destinations[obj]=dest
        label = f'{obj}:{dest}'
        name = ('positioned' if skill in ('place','table_place') else skill)+':'+label
        fact = self.graph.add('verification_fact', {'type':'skill_return_receipt',
            'historical_only':True, 'skill':skill, 'object':obj, 'destination':dest,
            'call_id':call, 'plan_id':plan, 'physical_effect_verified':False}, parents=[call])
        self.history[name] = fact
        effects = ['aligned:'+label] if skill in ('place','table_place') else (
            ['placed:'+label] if skill=='release' else ['safe_retreat'] if skill=='retreat' else [])
        self.expect(effects, fact)
        return fact

    def truth_map(self):
        return {k:self.get(k)['truth'] for k in self.ids}

    def complete(self):
        return all(self.get(k)['truth']=='TRUE' for k in self.goal_keys)

    def remaining_goals(self):
        return [k for k in self.goal_keys if self.get(k)['truth']!='TRUE']

    def repair_frontier(self):
        """Regress a remaining goal through explicit task-condition edges.

        Satisfied prerequisites are reused; missing ones become the repair
        frontier. This is goal regression only, never a robot action schedule.
        """
        def visit(key,path):
            if self.get(key)['truth']=='TRUE':return None
            if key in path:raise RuntimeError('Cyclic task condition contract')
            path=path+[key]
            node=self.graph.nodes.get(self.stage_goals.get(key),{})
            for prerequisite in node.get('data',{}).get('requires_conditions',[]):
                result=visit(prerequisite,path)
                if result:return result
            return key,path
        for key in self.goal_keys:
            result=visit(key,[])
            if result:return result
        return None,[]

    def check_event(self, m, check, status, fact):
        obj = m.held_object
        if obj and check in ('lift','hold_probe'):
            truth = 'TRUE' if status=='PASS' else 'UNKNOWN'
            for predicate in ('holding','lifted'):
                self.set(predicate+':'+obj,truth,evidence=[fact],epochs={'grasp':m.grasp_epoch},
                         reason='public_'+check)
        elif obj and check=='grasp':
            self.set('holding:'+obj,'UNKNOWN',evidence=[fact],epochs={'grasp':m.grasp_epoch},reason='closure_is_not_grasp_proof')

    def motion_effect(self, m, name):
        evidence = [m.last_call] if m.last_call else []
        self.set('safe_retreat','UNKNOWN',evidence=evidence,reason='robot_moved_since_retreat')
        if m.held_object and (m.closed or name=='open_gripper'):
            for obj,dest in self.stages:
                if obj==m.held_object:
                    self.set('placed:'+obj+':'+dest,'UNKNOWN',evidence=evidence,reason='object_manipulated')
                    self.set('aligned:'+obj+':'+dest,'UNKNOWN',evidence=evidence,reason='motion_expired_alignment')
        if name=='open_gripper' and m.held_object:
            for key in ('holding:','lifted:'):
                self.set(key+m.held_object,'UNKNOWN',evidence=evidence,reason='release_return_is_not_verified_effect')

    def api_source(self, source):
        node=self.graph.nodes[source];data=node['data'];typ=data.get('type');obj=data.get('object')
        if typ in ('surface','grasp','table_region','robot'):
            self.set('source:'+typ+':'+str(obj),'TRUE',evidence=[source],
                epochs={k:v for k,v in data.items() if k.endswith('_epoch')},
                reason='versioned_public_API_source')

    def refresh(self, m):
        """Recheck effects from public RGB-D only, without making physical calls.

        Missing surfaces give UNKNOWN. They never erase returned-action history
        or force automatic regrasp. Current alignment is not release permission;
        GroundedMonitor revalidates again at final physical dispatch.
        """
        evidence = [m.last_observation] if m.last_observation else []
        for obj,dest in self.stages:
            if dest=='held':continue
            label=obj+':'+dest;geom=m.geometry.get(obj)
            surface=m._surface(obj)
            epochs={'object':m.object_epochs.get(obj),'support':m.object_epochs.get(dest)}
            held=m.closed and m.held_object==obj
            if held:
                # Existing attachment checks are the sole source of held geometry.
                # Avoid invoking their check recorder from inside capture().
                continue
            if 'release:'+label not in self.history:continue
            if not reliable_top(surface,geom):
                # Reuse the existing CURRENT visible-face estimator. Historical
                # noncontradiction alone cannot certify a post-release effect.
                surface=m._support_surface(obj,allow_retained=False)
            # Select the current object estimate BEFORE defining the table
            # plane reference. Mixing the raw visible centroid with a refined
            # center creates a fictitious lateral placement error. This checks
            # post-release support only; release clearance remains separate.
            support=({'center':surface['center'],'hi':[0,0,m.table_z]}
                if dest=='table' and surface is not None and m.table_z is not None
                else m._support_surface(dest,allow_retained=False) if dest!='table' else None)
            check=place_metrics(surface,support,geom)
            truth={'PASS':'TRUE','FAIL':'FALSE','UNCERTAIN':'UNKNOWN'}[check['status']]
            self.set('placed:'+label,truth,evidence=evidence,epochs=epochs,
                reason='post_release_'+check['reason'],metrics=check)
        if m.closed and m.held_object:
            tcp=np.asarray(m.last_public.get('robot_cartesian_pos',[]))
            if len(tcp)<8 or not np.isfinite(tcp[7]):
                self.set('holding:'+m.held_object,'UNKNOWN',evidence=evidence,reason='missing_current_gripper')
            elif tcp[7]<.05:
                self.set('holding:'+m.held_object,'FALSE',evidence=evidence,reason='public_empty_jaws')

    def summary(self):
        return {'schema':'capx.task_conditions.v22','task':self.task,
            'conditions':[self.get(k) for k in self.ids], 'remaining_goals':self.remaining_goals(),
            'stages':self.stages, 'completed_receipts':dict(self.history),
            'complete_public_candidate':self.complete(), 'is_physical_reward':False}
