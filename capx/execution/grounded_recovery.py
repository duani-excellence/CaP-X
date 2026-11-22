"""Bounded, public-state skill recovery. Emits code for the EXISTING Sandbox.

Graph edges select reusable evidence, never the order of physical actions.
Recipes below order actions by the same skill/phase contracts as the actor API.
No environment, evaluator, network client or simulator object is accepted here.
"""
from __future__ import annotations
import copy
import time


class GroundedRecovery:
    def __init__(self, *, max_attempts=6, max_no_progress=2):
        if type(max_attempts) is not int or type(max_no_progress) is not int or min(max_attempts,max_no_progress)<1:
            raise ValueError('Recovery limits must be positive integers')
        self.max_attempts=max_attempts;self.max_no_progress=max_no_progress
        self.attempts=0;self.stalls={};self.pending=None;self.events=[]

    def _event(self,m,action,**fields):
        row={'action':action,**copy.deepcopy(fields)}
        self.events.append(row);m.graph._event('bounded_recovery',**{'recovery_action':action,**fields})
        return row

    def propose(self,m,*,execution_status):
        if self.pending is not None:raise RuntimeError('Complete prior recovery before proposing another')
        if execution_status not in ('ok','interrupted','runtime_error','invalid_program','policy_violation'):
            return None
        if m.terminal_reason or m.reconciliation_required or m.phase=='reconciliation_required':return None
        if self.attempts>=self.max_attempts or m.local_attempts>=m.max_episode_local:
            self._event(m,'fallback',reason='recovery_budget_exhausted');return None
        # Refresh public evidence before selecting retained sources; graph
        # invalidation caused by this capture must affect the selection itself.
        feedback=m.summary();intent=copy.deepcopy(getattr(m,'skill_intent',None) or m.last_place or {})
        held=m.held_object if m.closed else None
        support=None
        if intent.get('skill')=='place':
            source=m.graph.nodes.get(intent.get('inputs',{}).get('support_surface'),{})
            support=source.get('data',{}).get('object')
        if support is None and held=='red cube' and m.task in ('stack','restack'):support='green cube'
        kind=None
        if m.phase=='released':kind='fresh_retreat'
        elif m.phase=='holding_unverified' and m.closed:kind='verify_grasp'
        elif m.phase=='holding_verified' and held and m.task!='lift':kind='replace_placement'
        elif m.phase=='ready' and m.last_grasp_loss:kind='rebuild_after_grasp_loss'
        decision=None
        conditions=getattr(m,'conditions',None)
        if conditions and m.phase=='approached' and m.approached_object:
            kind='complete_grasp';held=m.approached_object
        if conditions and m.closed and conditions.get('holding:'+str(m.held_object))['truth']!='TRUE':
            kind='verify_grasp'
        if conditions and m.mode=='G1' and m.graph_options['selective_recovery'] and not m.conservative_fallback:
            from .graph_executive import GraphExecutive
            decision=GraphExecutive().decide(m)
            m.last_repair_decision=copy.deepcopy(decision)
            self._event(m,'condition_decision',decision=decision)
            kind=decision['kind'] if decision['recoverable'] else None
            obj=decision.get('object')
            if obj:held=obj
            dest=decision.get('destination')
            support=dest if dest in ('red cube','green cube') else None
        elif conditions and kind is None and m.phase=='ready':
            # Strong G0 keeps completed stage receipts, with the SAME legal
            # sensing/acquisition capabilities. It conservatively handles the
            # next incomplete suffix instead of graph-selecting a narrower one.
            for obj,dest in conditions.stages:
                goal='holding:'+obj if dest=='held' else 'placed:'+obj+':'+dest
                truth=conditions.get(goal)['truth']
                if truth=='TRUE':continue
                kind=('revalidate_effect' if truth=='UNKNOWN' and 'release:'+obj+':'+dest in conditions.history
                      else 'acquire_remaining')
                held=obj;support=dest if dest in ('red cube','green cube') else None
                break
        if kind is None:return None
        key=(kind,held or (m.last_grasp_loss or {}).get('object'),support)
        if self.stalls.get(key,0)>=self.max_no_progress:
            self._event(m,'fallback',reason='no_progress_limit',scope=list(key));return None
        selective=(m.mode=='G1' and m.graph_options['selective_recovery'] and not m.conservative_fallback)
        selected={};lines=[]
        def source(var,typ,obj,api,*,fresh=False):
            candidate=None
            key={'robot':'robot_read','table_region':'table_region'}.get(typ,typ+':'+str(obj))
            if selective and not fresh:
                for node in reversed(list(m.graph.nodes.values())):
                    d=node.get('data') or {};epoch=d.get('object_epoch')
                    if (node['kind']=='estimate' and node['valid'] and d.get('type')==typ
                        and (obj is None or d.get('object')==obj)
                        and (epoch is None or epoch==m.object_epochs.get(obj))
                        and node.get('key')==key and m.graph.latest.get(key)==node['id']
                        and d.get('frame')=='robot_base'
                        and (typ!='robot' or d.get('robot_epoch')==m.robot_epoch)):
                        candidate=node;break
            if candidate:
                selected[var]={'id':candidate['id'],'replacement_api':api,'type':typ,'object':obj}
                lines.append(f"{var} = recall_result({candidate['id']!r})")
            else:lines.append(f'{var} = {api}')
            return var+"['id']"
        def plan(skill,roles,params=None):
            inputs='{'+', '.join(repr(k)+': '+v for k,v in roles.items())+'}'
            lines.append(f"p = plan_skill({skill!r}, {inputs}, {params or {}!r})")
            lines.append('run_plan(p)')
        def retreat(*,fresh=False):
            r=source('r','robot',None,'read_robot()',fresh=fresh)
            plan('retreat',{'robot':r},intent.get('params') if intent.get('skill')=='retreat' else None)
        if kind=='fresh_retreat':
            retreat()
        elif kind=='release_retreat':
            # No cached PASS authorizes this call. open_gripper rechecks current
            # geometry, versions and permission at its final physical boundary.
            lines.append('open_gripper()');retreat(fresh=True)
        elif kind=='revalidate_effect':
            # Re-observation is evidence acquisition, never a repeated release
            # or retreat. Failure leaves the effect UNKNOWN and is bounded.
            lines.append(f'observe_surfaces({held!r})')
            if support:lines.append(f'observe_surfaces({support!r})')
            lines.append('verify_task()')
        elif kind=='verify_grasp':
            if m.lift_attempts==0:
                r=source('r','robot',None,'read_robot()');plan('lift',{'robot':r})
                # A successful first lift already verifies using the baseline.
                # If it interrupts, the next actor can use verify_hold instead.
            else:lines.append('verify_hold()')
        elif kind=='complete_grasp':
            lines.append('close_gripper()')
            r=source('r','robot',None,'read_robot()',fresh=True)
            plan('lift',{'robot':r})
        elif kind in ('rebuild_after_grasp_loss','acquire_remaining'):
            obj=held if kind=='acquire_remaining' else m.last_grasp_loss.get('object')
            if obj not in ('red cube','green cube'):return None
            lines.append('open_gripper()')
            # A lost object may have displaced its neighbor; fresh scene for BOTH
            # policies. Regrasp occurs only through existing sweep/approach gates.
            for name in (('red cube',) if m.task=='lift' else ('red cube','green cube')):
                # Revalidate an already calibrated, completed nonheld stage
                # through current surfaces; do not erase its upright calibration
                # by treating it as a new, unapproached object. Lost-grasp scene
                # reconstruction still uses fresh object localization for both.
                api='observe_surfaces' if kind=='acquire_remaining' and name!=obj and name in m.geometry else 'observe_object'
                lines.append(f'{api}({name!r})')
            g=source('g','grasp',obj,f'observe_grasp({obj!r})',fresh=True)
            plan('approach',{'grasp':g});lines.append('close_gripper()')
            r=source('r','robot',None,'read_robot()',fresh=True);plan('lift',{'robot':r})
        else:
            table=(intent.get('skill')=='table_place' or held=='green cube' and m.task=='restack')
            if not table and support not in ('red cube','green cube'):return None
            # A known occlusion permits at most one bounded upward observation
            # move per recipe. It is an explicit skill, not reuse of old descent.
            preconditions=feedback.get('recovery_preconditions',{})
            missing=preconditions.get('missing_surface_object')
            support_view_required=support is not None and missing==support
            if support_view_required:
                r=source('view_r','robot',None,'read_robot()',fresh=True)
                plan('lift',{'robot':r},{'dz':.08})
            changes_view=support_view_required
            if table:
                region=preconditions.get('region_id')
                revalidate=(isinstance(region,str) and preconditions.get('next_action')
                    =='reobserve_table_same_region_then_new_plan_or_stop')
                if revalidate:
                    # The prior bounds remain historical intent, never current
                    # permission. The existing API returns a NEW checked source
                    # and charges any view motion to the same episode budget.
                    changes_view=True
                    api=f'reobserve_table({region!r}, change_view=True)'
                else:api=f'observe_table({held!r})'
                t=source('t','table_region',held,api,fresh=True)
                roles={'table_region':t}
            else:
                if support not in m.geometry:lines.append(f'observe_object({support!r})')
                t=source('support','surface',support,f'observe_surfaces({support!r})',fresh=support_view_required)
                roles={'support_surface':t}
            # A pending view move expires pose handles even if they are valid
            # at recipe construction. Acquire them AFTER that operation.
            h=source('held','surface',held,f'observe_surfaces({held!r})',fresh=changes_view)
            r=source('r','robot',None,'read_robot()',fresh=changes_view)
            roles.update(held_surface=h,robot=r)
            skill='table_place' if table else 'place'
            plan(skill,roles,intent.get('params') if intent.get('skill')==skill else None)
            lines.append('open_gripper()');retreat(fresh=True)
        self.attempts+=1;m.local_attempts+=1;m.recoveries+=1
        recipe=dict(id=f'recovery-{self.attempts}',kind=kind,object=held or (m.last_grasp_loss or {}).get('object'),
            roles={'held':m.held_object if m.closed else None,'target_object':held,'support':support},code='\n'.join(lines),selected_sources=selected,
            policy='graph_selected_evidence' if selective else 'fresh_unexecuted_suffix',
            recovery_scope=feedback.get('recovery_scope',{}),diagnostic_roots=feedback.get('diagnosis',{}).get('roots',[]),
            initial_intent=intent,never_replay=list(m.call_history),budget_reserved=1)
        if conditions:
            recipe['condition_decision']=decision or dict(schema='capx.conservative_repair.v22',
                kind=kind,selected_actions=[kind],repair_goals=conditions.remaining_goals(),
                policy='phase_and_sequential_stage_suffix',failed_conditions=[],affected_nodes=[],
                preserved_nodes=[],reused_sources=[],new_plans=[],verification_results=[])
            recipe['condition_decision']['reused_sources']=[v['id'] for v in selected.values()]
            m.last_repair_decision=copy.deepcopy(recipe['condition_decision'])
        self.pending=dict(recipe=recipe,key=key,start=time.monotonic(),phase=m.phase,
            graph_sequence=len(m.graph.events),before_nodes=set(m.graph.nodes),before_calls=len(m.call_history),
            conditions_before=conditions.truth_map() if conditions else None)
        self._event(m,'proposed',**recipe)
        return copy.deepcopy(recipe)

    def complete(self,m,execution_status):
        if self.pending is None:return None
        p=self.pending;self.pending=None;r=p['recipe']
        plans=[n for i,n in m.graph.nodes.items() if i not in p['before_nodes'] and n['kind']=='plan']
        returned={e['plan_id'] for e in m.graph.events[p['graph_sequence']:] if e['action']=='skill_returned'}
        # Selection is NOT reuse. Only sources of a new plan that actually
        # returned are counted; an invalid/never-issued plan earns no benefit.
        used={sid for n in plans if n['id'] in returned for sid in n['data']['inputs'].values()}
        actual=[v for v in r['selected_sources'].values() if v['id'] in used]
        progress=(m.phase!=p['phase'] or bool(returned)) and execution_status=='ok'
        conditions=getattr(m,'conditions',None)
        if conditions:
            conditions.refresh(m)
            progress=(execution_status=='ok' and (m.phase!=p['phase'] or
                any(not k.startswith('source:') and v=='TRUE' and p['conditions_before'].get(k)!='TRUE'
                    for k,v in conditions.truth_map().items())))
        self.stalls[p['key']]=0 if progress else self.stalls.get(p['key'],0)+1
        outcome=self._event(m,'completed',id=r['id'],kind=r['kind'],execution_status=execution_status,
            phase_before=p['phase'],phase_after=m.phase,progress=progress,
            actual_reused_source_ids=[v['id'] for v in actual],
            avoided_acquisition_apis=[v['replacement_api'] for v in actual],
            avoided_calls_are_recipe_counterfactual_not_pair_measurement=True,
            new_plan_ids=[n['id'] for n in plans],returned_new_plan_ids=sorted(returned),
            verification_results=conditions.summary() if conditions else None,
            returned_api_calls=len(m.call_history)-p['before_calls'],seconds=time.monotonic()-p['start'])
        if conditions:
            m.last_repair_decision.update(new_plans=outcome['new_plan_ids'],
                verification_results=outcome['verification_results'],actual_reused_sources=outcome['actual_reused_source_ids'],
                progress=progress)
        return outcome
