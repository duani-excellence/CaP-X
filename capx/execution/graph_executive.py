"""Small deterministic condition-regression executive, not another agent.

Regress the first necessary task obligation onto existing skill recipes. Current
physical constraints (held/released/unknown effect) constrain admissible repair;
the dependency DAG chooses evidence and scope, never an action replay order.
"""
from __future__ import annotations
import copy


class GraphExecutive:
    def decide(self, m):
        c=m.conditions;g=m.graph
        result=dict(schema='capx.graph_repair.v22',recoverable=True,kind=None,
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
            elif m.task=='lift':
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
                    kind=('release_retreat' if c.get(aligned)['truth']=='TRUE'
                        and 'positioned:'+held+':'+dest in c.history else 'replace_placement')
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
            if kind=='replace_placement':
                prerequisites += ['source:surface:'+obj,'source:robot:None',
                    'source:table_region:'+obj if dest=='table' else 'source:surface:'+dest]
        elif kind=='verify_grasp':prerequisites=['holding:'+obj]
        elif kind=='fresh_retreat':prerequisites=['safe_retreat']
        elif kind=='revalidate_effect':prerequisites=[goal]
        elif kind=='acquire_remaining' and m.task=='restack' and obj=='red cube':
            prerequisites=['placed:green cube:table']
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
        repair=g.add('repair_goal',copy.deepcopy(result),parents=roots+[c.stage_goals.get(goal,c.goal_node)])
        result['decision_id']=repair
        return result

    @staticmethod
    def destination(m,obj):
        for name,dest in m.conditions.stages:
            if name==obj:return dest
        return None
