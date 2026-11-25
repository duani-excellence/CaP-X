"""Bounded public-evidence recovery for immutable, parameterized contracts.

No task dispatch, simulator or physical motion lives here. Both policies have
the same repair abilities; G1 slices evidence by the selected contract, G0
rechecks the unfinished suffix and retained task effects. Primitive admission
still refreshes live preconditions in either policy.
"""
import copy
import uuid


class ConditionPlanner:
    """Regress goals through declared producers; never order robot actions by graph IDs.

    UNKNOWN requests evidence first. FALSE requires a qualified producer. Evidence
    exhaustion is semantic (not frame IDs), and no producer means bounded failure.
    The returned frontier is a parent-owned work scope, not a motion permission.
    """
    def __init__(self, contracts):
        self.c = contracts
        self.exhausted = {}
        self._motion_epoch = None

    def motion_epoch(self, epoch):
        if epoch != self._motion_epoch:
            self.exhausted.clear()
            self._motion_epoch = epoch

    def exhaust(self, keys):
        for key in keys:
            self.exhausted[key] = self.c.truth(key)

    def costed_evidence(self,decision,features):
        """Choose a cheap frontier; co-observe free future dependency facts.

        Cost is the union of sensor/compute features supplied by the common
        skill registry. Never use task identity, physical truth or graph node
        count as utility. No positive incremental speculative grasp/IK work.
        """
        if decision['work']!='evidence':return decision
        c=self.c;goals=decision.get('goals',c.definition['goals'])
        # Consume a legal ready branch before collecting unrelated evidence.
        # Otherwise the all-goal frontier serially prepares every transient
        # plan before the first motion invalidates most of them. Regressing a
        # single goal still enforces every declared prerequisite and ordering.
        for goal in goals:
            ready=self.plan([goal])
            if ready['work']=='skill':
                return dict(ready,goals=goals,cost_policy='consume_ready_work_before_unrelated_evidence',
                            deferred=decision['keys'],estimated_cost=0.)
        specs=c.definition.get('condition_specs',{});future=[];visited=set()
        def visit(key):
            if key in visited or c.truth(key)=='TRUE':return
            visited.add(key)
            if c.truth(key)=='UNKNOWN' and specs.get(key,{}).get('observable') and self.exhausted.get(key)!='UNKNOWN':future.append(key)
            for row in c.definition['intents']:
                if key in row['effects']:
                    for req in row.get('requires',[]):visit(req)
        for goal in decision.get('goals',c.definition['goals']):visit(goal)
        requested=decision['keys'];costs={k:features(k) for k in set(future+requested)}
        def weight(value):return value['cost'] if isinstance(value,dict) else value
        def cost(key):return sum(weight(v) for v in costs[key].values())
        def transient(key):return any(isinstance(v,dict) and v.get('valid_until')=='motion' for v in costs[key].values())
        # Required sensory questions share one capture. Choosing the cheapest
        # question repeatedly increases total capture cost. Conversely, plans
        # which expire on motion cannot all be consumed before the first action.
        keys=[k for k in requested if not transient(k)]
        if not keys:keys=[min(requested,key=lambda k:(cost(k),requested.index(k)))]
        union={f:v for key in keys for f,v in costs[key].items()}
        for key in list(dict.fromkeys(requested+future)):
            if key not in keys and set(costs[key])<=set(union):keys.append(key)
        keys=sorted(keys,key=cost)
        return dict(decision,keys=keys,cost_policy='shared_sensor_bundle_with_motion_lifetime_and_free_future_closure',
            original_keys=requested,features=union,estimated_cost=sum(weight(v) for v in union.values()),
            avoided_incremental_cost=0.,deferred=[k for k in requested if k not in keys])

    def plan(self, goals=None):
        c = self.c
        goals = list(goals if goals is not None else c.definition['goals'])
        specs = c.definition.get('condition_specs', {})
        blocked = []
        preserved = {k:n for k,n in c.conditions.items() if c.truth(k)=='TRUE'}
        if c.quarantined:
            return dict(work='blocked',allowed_intents=[],keys=[],blocked=[dict(reason='quarantined')],preserved=preserved)

        def regress(key, path):
            truth = c.truth(key)
            if truth == 'TRUE': return dict(work='satisfied')
            if key in path:
                blocked.append(dict(condition=key,truth=truth,reason='dependency_cycle'))
                return dict(work='blocked')
            if truth == 'UNKNOWN':
                if specs.get(key,{}).get('observable') and self.exhausted.get(key)!=truth:
                    return dict(work='evidence',keys=[key],allowed_intents=[],condition=key)
                blocked.append(dict(condition=key,truth=truth,reason='evidence_unavailable'))
                return dict(work='blocked')
            producers = [r for r in c.definition['intents']+getattr(c,'producer_library',[]) if key in r['effects']]
            if not producers:
                blocked.append(dict(condition=key,truth=truth,reason='no_producer'))
                return dict(work='blocked')
            choices=[]
            for row in producers:
                missing=[k for k in row.get('requires',[]) if c.truth(k)!='TRUE']
                needs=[regress(k,path+[key]) for k in missing]
                # A proven unavailable shared prerequisite cuts this branch. Do
                # not photograph its other prerequisites without a recovery route.
                if any(n['work']=='blocked' for n in needs):continue
                if needs:
                    choices.extend(n for n in needs if n['work']!='satisfied');continue
                if not c.order_satisfied(row):
                    blocked.append(dict(condition=key,truth=truth,reason='order_unsatisfied',intent=row['id']));continue
                return dict(work='skill',keys=[],allowed_intents=[row['id']],condition=key,
                    new_plan_required=[row['id']] if row['id'] in c.completed or row['id'] in c.stopped else [])
            if choices:
                evidence=[k for n in choices if n['work']=='evidence' for k in n['keys']]
                if evidence:return dict(work='evidence',keys=list(dict.fromkeys(evidence)),allowed_intents=[],condition=key)
                return choices[0]
            return dict(work='blocked')

        candidates=[regress(goal,[]) for goal in goals if c.truth(goal)!='TRUE']
        evidence=[k for work in candidates if work['work']=='evidence' for k in work['keys']]
        # One public frame can satisfy multiple independent frontier questions.
        # It is still a selective goal cut, not a whole-suffix recheck.
        if evidence:
            return dict(work='evidence',keys=list(dict.fromkeys(evidence)),allowed_intents=[],
                blocked=blocked,preserved=preserved,goals=goals)
        ready=next((w for w in candidates if w['work']=='skill'),None)
        if ready:return dict(ready,blocked=blocked,preserved=preserved,goals=goals)
        return dict(work='complete' if not candidates else 'blocked', keys=[],allowed_intents=[],
            blocked=blocked,preserved=preserved,goals=goals)



class ContractRecovery:
    def __init__(self,contracts,broker,*,group,max_repairs=6,max_same_frontier=2,emit=None):
        if group not in ('G0','G1'):raise ValueError('Unknown policy')
        self.c=contracts;self.broker=broker;self.group=group
        self.max_repairs=max_repairs;self.max_same_frontier=max_same_frontier
        self.repairs=0;self.frontiers={};self.emit=emit or (lambda row:None)

    def scope(self,intent):
        c=self.c;row=c.intents[intent]
        if self.group=='G0':
            keys=[k for i,r in c.intents.items() if i not in c.completed for k in r.get('requires',[])]+c.definition['goals']
        else:
            # Effects distinguish missing knowledge from genuinely unfinished work.
            keys=row.get('requires',[])+row['effects']
            keys += [k for k in c.definition['goals'] if k in c.conditions and c.truth(k)!='TRUE']
        return list(dict.fromkeys(keys))

    def refresh_for(self,intent):
        if self.c.quarantined:raise ValueError('Physical effect quarantined')
        before=dict(self.c.conditions);scope=self.scope(intent)
        rows=self.broker.refresh(scope)
        preserved={k:v for k,v in before.items() if k not in scope and self.c.truth(k)=='TRUE'}
        result=dict(intent=intent,scope=scope,observed=rows,preserved=preserved,
                    policy=self.group,completed_calls=list(self.c.completed))
        self.emit(dict(kind='recovery_evidence',**result));return result

    def repair(self,intent):
        c=self.c
        if c.quarantined:raise ValueError('Physical effect quarantined')
        row=c.intents[intent]
        if all(c.truth(k)=='TRUE' for k in row['effects']):raise ValueError('Effect already verified; no repeated physical call')
        if intent not in c.stopped and intent not in c.completed:raise ValueError('Use the existing unexecuted intent')
        if self.repairs>=self.max_repairs:raise ValueError('Repair budget exhausted')
        original=row.get('repair_of',intent)
        # Truth/phase frontier, not source IDs: a new frame alone is no progress.
        signature=(original,tuple((k,c.truth(k)) for k in sorted(set(row.get('requires',[])+row['effects']))))
        count=self.frontiers.get(signature,0)
        if count>=self.max_same_frontier:raise ValueError('No condition progress; repeated repair stopped')
        self.frontiers[signature]=count+1;self.repairs+=1
        new=copy.deepcopy(row);new.update(id='repair-'+uuid.uuid4().hex,repair_of=original)
        c.register(new);self.emit(dict(kind='repair_intent',old=intent,new=new['id'],frontier=signature))
        return new['id']

