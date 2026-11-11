import ast
from pathlib import Path
from types import SimpleNamespace


def test_dataset_actually_resets_each_seed():
    source=Path('capx/cli/prepare_verl_dataset.py').read_text()
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_rows')
    calls=[]
    class Fake:
        oracle_code='pass'
        def reset(self,*,seed):
            calls.append(seed)
            return self._get_observation(), {'initial_state_hash':str(seed)}
        def _get_observation(self):
            return {'full_prompt':[{'role':'user','content':'go'}]}
        def close(self): pass
    namespace={'get_exec_env':lambda x:lambda cfg:Fake(),'get_config':lambda x:SimpleNamespace(privileged=False)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'dataset','exec'),namespace)
    rows=namespace['_generate_rows']([0,2],split='test',data_source='mock',seed_base=70)
    assert calls==[70,72]
    assert [r['extra_info']['initial_state_hash'] for r in rows]==['70','72']
