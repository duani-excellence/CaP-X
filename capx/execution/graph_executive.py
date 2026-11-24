"""Small deterministic condition-regression executive, not another agent.

Regress the first necessary task obligation onto existing skill recipes. Current
physical constraints (held/released/unknown effect) constrain admissible repair;
the dependency DAG chooses evidence and scope, never an action replay order.
"""
from __future__ import annotations
import copy


class GraphExecutive:
    def repair_contract(self, contracts, planner):
        """Select a condition-regressed frontier; the parent enforces its work scope.

        The stateful planner owns bounded evidence attempts across actor lifetimes.
        No task/environment identity or evaluator data is available here.
        """
        if planner.c is not contracts:raise ValueError('Planner contract ownership mismatch')
        return planner.plan()

    def next_contract(self, contracts, *, excluded=()):
        """Select declared work by conditions and resource constraints, not names.

        This entry is used by non-cube adapters. A returned intent is only a
        proposal: ContractGraph.admit plus the parent primitive verifier own
        permission. Unknown effects never get silently retried.
        """
        if contracts.quarantined:return dict(work='stop',reason='unknown_physical_effect')
        if contracts.complete():return dict(work='verify_task',reason='public_goals_satisfied')
        evidence_work=[]
        for key,row in contracts.intents.items():
            if key in excluded:continue
            if key in contracts.completed:continue
            if not contracts.order_satisfied(row):continue
            if any(r in contracts.locks for r in row.get('resources',[])):continue
            # Already physically verified effects need no repeated action.
            if all(contracts.truth(k)=='TRUE' for k in row['effects']):continue
            missing=[k for k in row.get('requires',[]) if contracts.truth(k)!='TRUE']
            decision=dict(work='evidence' if missing else 'skill',intent=key,
                          missing=missing,skill=row['skill'])
            if not missing:return decision
            evidence_work.append(decision)
        if evidence_work:return min(evidence_work,key=lambda d:len(d['missing']))
        if excluded:return dict(work='stop',reason='remaining_obligations_lack_progress_or_capability',excluded=list(excluded))
        repair=contracts.repair_obligation()
        if repair is not None:
            row=contracts.intents[repair]
            missing=[k for k in row.get('requires',[]) if contracts.truth(k)!='TRUE']
            return dict(work='evidence' if missing else 'skill',intent=repair,
                        missing=missing,skill=row['skill'],repair_of=row['repair_of'])
        return dict(work='stop',reason='no_admissible_declared_work')

    def decide(self, m):
        c=m.conditions;g=m.graph
        result=dict(schema='capx.graph_repair.v23',recoverable=True,kind=None,
            failed_conditions=[],affected_nodes=[],preserved_nodes=[],repair_goals=[],
            selected_actions=[],reused_sources=[],new_plans=[],verification_results=[],
            remaining_goals=c.remaining_goals(),never_replay=list(m.call_history))
        if m.reconciliation_required or m.terminal_reason:
            return dict(result,recoverable=False,reason='unknown_effect_or_terminal',
                        failed_conditions=['known_physical_effect'])
        held=m.held_object if m.closed else None
        obj=held;dest=None;goal=None;kind=None
        if m.phase=='released':
            obj=(m.last_place or {}).get('object',m.held_object)
            dest=self.destination(m,obj);goal='safe_retreat';kind='fresh_retreat'
        elif m.phase=='approached' and m.approached_object:
            obj=m.approached_object;dest=self.destination(m,obj)
            goal='holding:'+obj;kind='complete_grasp'
        elif held:
            dest=self.destination(m,held);goal='holding:'+held
            if dest is None:return dict(result,recoverable=False,reason='held_role_outside_known_task_recipe')
            hold=c.get(goal)['truth']
            if hold=='FALSE':
                # Let the existing pre-action public-jaw gate commit grasp loss;
                # a closed command flag is never permission to regrasp blindly.
                kind='verify_grasp'
            elif hold!='TRUE':kind='verify_grasp'
            elif dest=='held':
                if c.get('lifted:'+held)['truth']!='TRUE':kind='verify_grasp'
            else:
                goal='placed:'+held+':'+dest
                stage=g.nodes.get(c.stage_goals.get(goal),{}).get('data',{})
                missing=[key for key in stage.get('requires_conditions',[]) if c.get(key)['truth']!='TRUE']
                if missing:
                    failed=missing[0];truth=c.get(failed)['truth']
                    prerequisite=g.nodes[c.stage_goals[failed]]['data']
                    if truth=='FALSE':
                        return dict(result,recoverable=False,reason='prior_stage_broken_while_holding_other_object_requires_new_plan',
                            failed_conditions=missing,repair_goals=missing,
                            preserved_nodes=[c.get('holding:'+held)['id']],
                            diagnostic_roots=[c.get(key)['id'] for key in missing],
                            affected_nodes=[i for i,n in g.nodes.items() if n['kind']=='plan' and not n['valid']])
                    obj=prerequisite['object'];dest=prerequisite['destination']
                    goal=failed;kind='revalidate_effect'
                aligned='aligned:'+held+':'+dest
                # A CURRENT check and returned positioning receipt allow release
                # without repeating place; release's final gate remains mandatory.
                if not missing:
                    positioned='positioned:'+held+':'+dest in c.history
                    kind=('release_retreat' if c.get(aligned)['truth']=='TRUE' and positioned else
                          'verify_placement' if c.get(aligned)['truth']=='UNKNOWN' and positioned else 'replace_placement')
                    # A surface capture cannot establish the table skill's
                    # instance/robot/footprint clearance contract. Re-enter its
                    # targeted evidence recipe and compile a fresh plan; the
                    # controller skips already-converged physical segments.
                    if dest=='table' and c.get('clearance:'+held+':'+dest)['truth']!='TRUE':
                        kind='replace_placement'
        else:
            goal,path=c.repair_frontier()
            result['regression_path']=path
            node=g.nodes.get(c.stage_goals.get(goal),{})
            if node:
                obj=node['data']['object'];dest=node['data']['destination'];truth=c.get(goal)['truth']
                if truth=='UNKNOWN' and 'release:'+obj+':'+dest in c.history:
                    kind='revalidate_effect'
                else:kind='acquire_remaining'
        if kind is None:return dict(result,reason='no_known_repair_obligation')
        # Explicitly regress this goal onto physical conditions/data, not onto
        # the last Python source line or a fixed full suffix.
        prerequisites=[]
        if kind in ('replace_placement','release_retreat'):
            prerequisites=['holding:'+obj,'aligned:'+obj+':'+dest]
            if dest=='table':prerequisites.append('clearance:'+obj+':table')
            if kind=='replace_placement':
                prerequisites += ['source:surface:'+obj,'source:robot:None',
                    'source:table_region:'+obj if dest=='table' else 'source:surface:'+dest]
        elif kind=='verify_grasp':prerequisites=['holding:'+obj]
        elif kind=='fresh_retreat':prerequisites=['safe_retreat']
        elif kind=='revalidate_effect':prerequisites=[goal]
        elif kind=='verify_placement':prerequisites=['aligned:'+obj+':'+dest]
        elif kind=='acquire_remaining':prerequisites=c.required('approach',obj)
        failed=[k for k in dict.fromkeys([goal]+prerequisites) if c.get(k)['truth']!='TRUE']
        roots=[c.get(k)['id'] for k in failed if c.get(k)['id']]
        roots=list(dict.fromkeys(roots+[i for i in m.roots if i in g.nodes]))
        relevant=g.ancestors(roots) if roots else []
        impacted=set(roots);pending=list(roots)
        while pending:
            for child in g._propagating_children[pending.pop()]:
                if child not in impacted:impacted.add(child);pending.append(child)
        affected=[i for i in impacted if g.nodes[i]['kind'] in ('condition','plan','check')
                  and not (g.nodes[i].get('data') or {}).get('execution')=='consumed']
        preserved=[c.get(k)['id'] for k in c.ids if c.get(k)['truth']=='TRUE']
        result.update(kind=kind,object=obj,destination=dest,repair_goals=[goal],
            failed_conditions=failed,affected_nodes=sorted(affected),preserved_nodes=preserved,
            selected_actions=[kind],diagnostic_roots=roots,upstream_evidence=relevant,
            retained_task_stages=[f'{o}:{d}' for o,d in c.stages if c.get(f'placed:{o}:{d}')['truth']=='TRUE'],
            reason='regress_unsatisfied_goal_subject_to_current_physical_constraints')
        result['work_kind']=('acquire_evidence' if kind in ('verify_grasp','verify_placement','revalidate_effect')
            else 'continue_verified_suffix' if kind in ('release_retreat','fresh_retreat') else 'physical_repair')
        next_skill={'acquire_remaining':'approach','complete_grasp':'close_gripper',
            'fresh_retreat':'retreat','release_retreat':'open_gripper',
            'replace_placement':'table_place' if dest=='table' else 'place'}.get(kind)
        result['future_intent_id']=c.intent(next_skill,obj,dest) if next_skill else None
        repair=g.add('repair_goal',copy.deepcopy(result),parents=roots+[c.stage_goals.get(goal,c.goal_node)])
        result['decision_id']=repair
        return result

    @staticmethod
    def destination(m,obj):
        for name,dest in m.conditions.stages:
            if name==obj:return dest
        return None
