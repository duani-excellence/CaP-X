"""Bounded public-evidence recovery for immutable, parameterized contracts.

No task dispatch, simulator or physical motion lives here. Both policies have
the same repair abilities; G1 slices evidence by the selected contract, G0
rechecks the unfinished suffix and retained task effects. Primitive admission
still refreshes live preconditions in either policy.
"""
import copy
import uuid


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

