"""Fresh-process URL initialization checks; no service/model/env construction."""
import os
from pathlib import Path
import subprocess
import sys


def run_fresh(script):
    import capx
    repo=Path(capx.__file__).resolve().parent.parent
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=repo,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_import_graph_runner_does_not_cache_service_defaults():
    run_fresh('''import sys
import capx.experiments.graph_feedback
assert 'capx.envs' not in sys.modules
assert 'capx.integrations.vision.sam3' not in sys.modules
assert 'capx.integrations.vision.graspnet' not in sys.modules
assert 'capx.integrations.motion.pyroki' not in sys.modules
''')


def test_explicit_graph_cli_urls_apply_before_first_heavy_import():
    run_fresh('''import os,tempfile
from pathlib import Path
for name in ('SAM3','GRASPNET','PYROKI'):os.environ[name+'_SERVICE_URL']='http://old.invalid:1'
from capx.experiments import graph_feedback as runner
class StopBeforeRequests(Exception):pass
def inspect_budget(*args,**kwargs):
    from capx.integrations.vision import sam3,graspnet
    from capx.integrations.motion import pyroki
    assert sam3.SERVICE_URL=='http://sam3.invalid:19201'
    assert graspnet.SERVICE_URL=='http://graspnet.invalid:19202'
    assert pyroki.DEFAULT_URL=='http://pyroki.invalid:19203'
    raise StopBeforeRequests
runner.ModelBudget=inspect_budget
runner.archive_sources=lambda *args:{}
runner.subprocess.check_output=lambda *args,**kwargs:'mock-head'
with tempfile.TemporaryDirectory() as temp:
    try:
        runner.main(['--output',str(Path(temp)/'run'),'--sam3-url','http://sam3.invalid:19201',
            '--graspnet-url','http://graspnet.invalid:19202','--pyroki-url','http://pyroki.invalid:19203'])
    except StopBeforeRequests:pass
    else:raise AssertionError('Expected early exit before requests or simulation')
''')
