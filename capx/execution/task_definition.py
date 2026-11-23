"""Declarative task bindings. Only this adapter knows benchmark task names.

Definitions contain goal dependencies, not executable robot recipes. A caller
may supply another definition without changing the executive or recovery code.
"""
import copy


def benchmark_definition(name):
    objects={'red cube':dict(sensor_key='red',perception_name='red cube'),
             'green cube':dict(sensor_key='green',perception_name='green cube')}
    specifications={
        'stack':([('red cube','green cube',[])],['placed:red cube:green cube','safe_retreat']),
        'lift':([('red cube','held',[])],['holding:red cube','lifted:red cube']),
        'restack':([('green cube','table',[]),('red cube','green cube',['placed:green cube:table'])],
                   ['placed:red cube:green cube','safe_retreat']),
    }
    if name not in specifications:raise ValueError('Unknown benchmark definition')
    stages,goals=specifications[name]
    if name=='lift':objects.pop('green cube')
    return validate_definition(dict(name=name,objects=objects,goals=goals,subgoals=[
        dict(condition='holding:'+o if d=='held' else 'placed:'+o+':'+d,
             object=o,destination=d,requires=r) for o,d,r in stages]))


def validate_definition(value):
    d=copy.deepcopy(value)
    if not isinstance(d,dict) or not isinstance(d.get('objects'),dict) or not d['objects']:
        raise ValueError('Object bindings required')
    if any(not isinstance(o,str) or ':' in o or not o for o in d['objects']):raise ValueError('Invalid object identity')
    rows=d.get('subgoals',[]);keys=[r['condition'] for r in rows]
    if not rows or len(keys)!=len(set(keys)):raise ValueError('Unique subgoals required')
    for r in rows:
        obj,dest=r['object'],r['destination']
        if obj not in d['objects'] or dest not in {*d['objects'],'held','table'} or dest==obj:
            raise ValueError('Unknown or invalid role binding')
        expected='holding:'+obj if dest=='held' else 'placed:'+obj+':'+dest
        if r['condition']!=expected or not set(r.get('requires',[]))<=set(keys):raise ValueError('Invalid goal contract')
    allowed=set(keys)|{'safe_retreat'}|{'lifted:'+o for o in d['objects']}
    if not d.get('goals') or not set(d['goals'])<=allowed:raise ValueError('Unsupported goal predicate')
    bykey={r['condition']:r for r in rows}
    def walk(key,path):
        if key in path:raise ValueError('Cyclic goal dependencies')
        for p in bykey[key].get('requires',[]):walk(p,path+[key])
    for k in keys:walk(k,[])
    # Same object can have several destinations, but selection must be explicit
    # through goal edges. This first adapter rejects ambiguous live bindings.
    if len({r['object'] for r in rows})!=len(rows):raise ValueError('Ambiguous multiple live destinations')
    return d


def resolve_definition(value):
    return benchmark_definition(value) if isinstance(value,str) else validate_definition(value)
