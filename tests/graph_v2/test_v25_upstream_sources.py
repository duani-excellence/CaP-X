import hashlib
import importlib.util
from pathlib import Path
import pytest


def module(name):
    p=Path(__file__).resolve().parents[2]/'scripts'/'local'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_existing_upstream_checkout_is_read_only_and_version_checked(monkeypatch,tmp_path):
    m=module('fetch_upstream_tasks');calls=[]
    def output(command,**kw):
        calls.append(command)
        return 'abc\n' if command[-1]=='HEAD' else ''
    monkeypatch.setattr(m.subprocess,'check_output',output)
    assert m.verify_checkout(tmp_path,'abc')['commit']=='abc'
    with pytest.raises(ValueError,match='differs'):m.verify_checkout(tmp_path,'other')
    assert all('checkout' not in c and 'reset' not in c for c in calls)


def test_modified_upstream_source_is_not_mistaken_for_pinned_commit(monkeypatch,tmp_path):
    m=module('fetch_upstream_tasks')
    monkeypatch.setattr(m.subprocess,'check_output',lambda c,**kw:'abc\n' if c[-1]=='HEAD' else ' M physics.py\n')
    with pytest.raises(ValueError,match='modified'):m.verify_checkout(tmp_path,'abc')


def test_reference_dataset_hash_mismatch_preserves_existing_bytes(tmp_path):
    m=module('fetch_upstream_tasks');p=tmp_path/'reference.hdf5';p.write_bytes(b'old data')
    with pytest.raises(ValueError,match='hash mismatch'):m.verify_reference(p,'0'*64)
    assert p.read_bytes()==b'old data'
    assert m.verify_reference(p,hashlib.sha256(b'old data').hexdigest())['bytes']==8
