"""Small parent-owned execution dependency DAG; contains no robot or model access.

A completed call is an immutable historical fact, not a reusable instruction.
Invalidation traverses it to future work, but never changes it back to pending.
Only explicit version updates invalidate dependencies; routine observations may
be registered with ``invalidate=False``. Data should be compact public evidence
or IDs pointing to separately stored observations and execution logs.
"""
from __future__ import annotations

import copy
import json


_IMMUTABLE = {'program', 'capability', 'verification_fact'}
_WORK = {'observation', 'estimate', 'plan', 'check'}
_EDGE_KINDS = {'data', 'precondition', 'attestation'}


def _plain(value):
    def convert(item):
        if hasattr(item, 'tolist'):
            return item.tolist()
        raise TypeError('Graph data must be JSON serializable: ' + type(item).__name__)
    # Own the data, reject NaN, and make snapshots independent of mutable callers.
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


class DependencyGraph:
    def __init__(self, mode='G1', emit=None):
        if mode not in ('G0', 'G1'):
            raise ValueError('Graph mode must be G0 or G1')
        self.mode = mode
        self.emit = emit or (lambda event: None)
        self.nodes = {}
        self.edges = []
        self.latest = {}
        self.events = []
        self._versions = {}
        self._children = {}
        self._parents = {}
        self._propagating_children = {}

    def _event(self, action, **fields):
        event = _plain(dict(sequence=len(self.events) + 1, action=action, **fields))
        self.events.append(event)
        self.emit(copy.deepcopy(event))

    def _ids(self, ids):
        ids = [ids] if isinstance(ids, str) else list(ids)
        for node in ids:
            if node not in self.nodes:
                raise KeyError('Unknown graph node: ' + str(node))
        return ids

    def _parent_edges(self, parents):
        result = []
        for parent in parents:
            source, kind = (parent, 'data') if isinstance(parent, str) else parent
            if source not in self.nodes:
                raise KeyError('Unknown parent graph node: ' + str(source))
            if kind not in _EDGE_KINDS:
                raise ValueError('Unsupported dependency edge kind: ' + str(kind))
            if (source, kind) not in result:
                result.append((source, kind))
        return result

    def add(self, kind, data=None, parents=(), key=None, state='pending'):
        parents = self._parent_edges(parents)
        data = _plain(data)
        if key is not None and not isinstance(key, str):
            raise TypeError('Version keys must be strings')
        if state not in ('pending', 'done', 'invalid'):
            raise ValueError('Unsupported graph state: ' + str(state))
        node_id = 'n' + str(len(self.nodes) + 1).zfill(6)
        valid = state != 'invalid' and all(self.nodes[p]['valid'] for p, _ in parents)
        # Completed calls are factual records; their historical dependencies may
        # no longer describe the current world, but completion is not undone.
        if kind in _IMMUTABLE or (kind == 'call' and state == 'done'):
            valid = True
        elif not valid:
            state = 'invalid'
        version = None
        if key is not None:
            version = self._versions.get(key, 0) + 1
            self._versions[key] = version
            self.latest[key] = node_id
        node = dict(id=node_id, kind=kind, key=key, version=version, state=state,
                    valid=valid, data=data, invalidated_by=[])
        self.nodes[node_id] = node
        self._children[node_id] = []
        self._parents[node_id] = []
        self._propagating_children[node_id] = []
        for source, edge_kind in parents:
            self.edges.append(dict(source=source, target=node_id, kind=edge_kind))
            self._children[source].append(node_id)
            self._parents[node_id].append(source)
            # Historical verification remains provenance, not a current pose
            # dependency. Explicit physical condition updates still propagate.
            if edge_kind != 'attestation':self._propagating_children[source].append(node_id)
        self._event('add', node=node, parents=[dict(source=p, kind=k) for p, k in parents])
        return node_id

    def version(self, key, kind, data=None, parents=(), reason='update', invalidate=True):
        # Validate before mutating existing nodes or consuming a version number.
        parents = self._parent_edges(parents)
        data = _plain(data)
        if not isinstance(key, str):
            raise TypeError('Version keys must be strings')
        previous = self.latest.get(key)
        old_versions = [n['id'] for n in self.nodes.values() if n['key'] == key and n['valid']]
        changed = []
        if invalidate and old_versions:
            changed = self.invalidate(old_versions, reason, conservative=self.mode == 'G0')
        current = self.add(kind, data, parents, key=key)
        self._event('version', key=key, previous=previous, current=current,
                    reason=reason, invalidation_enabled=bool(invalidate), invalidated=changed)
        return current

    def invalidate(self, ids, reason, conservative=False):
        roots = self._ids(ids)
        pending = list(roots)
        if conservative:
            pending.extend(n['id'] for n in self.nodes.values() if n['kind'] in _WORK)
        visited = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self._propagating_children[current])
        changed = []
        for node_id, node in self.nodes.items():
            if node_id not in visited or node['kind'] in _IMMUTABLE:
                continue
            if node['kind'] == 'call' and node['state'] == 'done':
                continue
            if node['valid']:
                node.update(valid=False, state='invalid')
                node['invalidated_by'].append(str(reason))
                changed.append(node_id)
        self._event('invalidate', roots=roots, reason=str(reason),
                    conservative=bool(conservative), invalidated=changed)
        return changed

    def require_valid(self, node_id):
        self._ids([node_id])
        node = self.nodes[node_id]
        if not node['valid']:
            raise RuntimeError('Graph node invalid; recompute or revalidate before execution: ' + node_id)
        return node

    def mark_done(self, node_id):
        node = self.require_valid(node_id)
        if node['state'] == 'done':
            raise RuntimeError('Graph node already completed; physical actions must not replay: ' + node_id)
        node['state'] = 'done'
        self._event('done', node_id=node_id)

    def returned_call(self, node_id):
        """Record an actual returned call, even if its input became obsolete.

        This records history, not permission to execute an invalid instruction.
        The trusted broker calls it only after the physical/API callback returns;
        invalid descendants remain invalid until explicitly recomputed.
        """
        self._ids([node_id])
        node = self.nodes[node_id]
        if node['kind'] != 'call':
            raise ValueError('returned_call requires a call node')
        if node['state'] == 'done':
            raise RuntimeError('Graph call already completed; physical actions must not replay: ' + node_id)
        previous_state, previous_valid = node['state'], node['valid']
        node.update(state='done', valid=True)
        self._event('returned_call', node_id=node_id, previous_state=previous_state,
                    previous_valid=previous_valid, state='done', valid=True)

    def consume_plan(self, node_id):
        """Reserve one execution attempt without claiming its physical outcome."""
        node = self.require_valid(node_id)
        if node['kind'] != 'plan':
            raise ValueError('consume_plan requires a plan node')
        data = node['data']
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError('Plan data must be a mapping or None')
        if data.get('execution') == 'consumed':
            raise RuntimeError('Graph plan already consumed; create and validate a new plan: ' + node_id)
        data['execution'] = 'consumed'
        node['data'] = data
        self._event('consume_plan', node_id=node_id, execution='consumed')

    def ancestors(self, ids):
        pending = self._ids(ids)
        visited = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self._parents[current])
        # Parents always predate children: creation order is a topological order.
        return [node_id for node_id in self.nodes if node_id in visited]

    def diagnose(self, ids, max_nodes=40):
        roots = self._ids(ids)
        if max_nodes < 1:
            raise ValueError('max_nodes must be positive')
        relevant = self.ancestors(roots)
        # Keep failure roots first even when their ancestors/history is huge.
        failures=[n for n in reversed(relevant) if self.nodes[n]['kind']=='check'
                  and (self.nodes[n]['data'] or {}).get('status') in ('FAIL','UNCERTAIN')]
        chosen = list(dict.fromkeys(roots + failures + list(reversed(relevant))))[:max_nodes]
        selected = relevant if len(relevant)<=max_nodes else chosen
        selected_set = set(selected)
        evidence_invalid = [n for n in relevant if not self.nodes[n]['valid']]
        # Diagnosis is a backward evidence slice, but recovery must also account
        # for the affected future suffix (for example place -> retreat). Start
        # only from invalid prerequisites of these roots, not every old failure.
        pending = list(evidence_invalid)
        affected = set()
        while pending:
            current = pending.pop()
            if current in affected:
                continue
            affected.add(current)
            pending.extend(self._children[current])
        invalid = [n for n in self.nodes if n in affected and not self.nodes[n]['valid']]
        preserved = [n for n, node in self.nodes.items() if node['valid']]
        completed = [n for n, node in self.nodes.items() if node['kind'] == 'call' and node['state'] == 'done']
        from .diagnostic_context import bounded
        cap=lambda values: values[-max_nodes:]
        def by_kind(kinds):
            return cap([n for n in invalid if self.nodes[n]['kind'] in kinds])
        nodes=[]
        for n in selected:
            node=copy.deepcopy(self.nodes[n])
            node['data'],stats=bounded(node['data'],max_chars=1200,max_items=8,max_string=240)
            if stats['truncated']:node['data_truncation']=stats
            node['invalidated_by']=cap(node['invalidated_by'])
            nodes.append(node)
        result = dict(schema='capx.execution_dependencies.v1', mode=self.mode, roots=roots[:max_nodes],
                      nodes=nodes,
                      edges=[e for e in self.edges if e['source'] in selected_set and e['target'] in selected_set][:max_nodes*4],
                      invalidated=cap(invalid), evidence_invalidated=cap(evidence_invalid), preserved=cap(preserved),
                      completed_calls=cap(completed), truncated=any(len(v)>max_nodes for v in (relevant,roots,invalid,preserved,completed)),
                      recovery_scope=dict(impacted=cap(invalid), preserved=cap(preserved),
                                          scope_basis='invalid evidence ancestors and their invalid downstream suffix; completed calls never replay',
                                          reobserve=by_kind({'observation', 'estimate'}),
                                          recompute=by_kind({'plan'}), revalidate=by_kind({'check'}),
                                          blocked_calls=by_kind({'call'}),
                                          never_replay=cap(completed),
                                          policy='all_unexecuted_suffix' if self.mode == 'G0' else 'affected_dependencies'))
        result['diagnostic_stats']=dict(graph_nodes=len(self.nodes),relevant_nodes=len(relevant),
            roots=len(roots),invalidated=len(invalid),preserved=len(preserved),completed_calls=len(completed),
            selected_nodes=len(nodes),max_nodes=max_nodes,reason='bounded_recovery_slice_full_history_archived',
            execution_consults_full_graph=True)
        return copy.deepcopy(result)

    def snapshot(self):
        return copy.deepcopy(dict(schema='capx.execution_dependencies.v1', mode=self.mode,
                                  nodes=self.nodes, edges=self.edges, latest=self.latest,
                                  versions=self._versions, events=self.events))
