"""Opt-in regression against the configured task renderer, not an empty GL context.

Set CAPX_DEPTH_RUNTIME_CONFIG/SNAPSHOT to archived public experiment inputs.
No LLM/perception HTTP request or robot API is issued; reset settling is replayed.
"""
import json
import os
from pathlib import Path
import numpy as np
import pytest


def _state(g):
    enums=('GL_READ_FRAMEBUFFER_BINDING','GL_DRAW_FRAMEBUFFER_BINDING',
        'GL_RENDERBUFFER_BINDING','GL_VIEWPORT','GL_MATRIX_MODE','GL_CURRENT_PROGRAM',
        'GL_DEPTH_FUNC','GL_CLIP_ORIGIN','GL_CLIP_DEPTH_MODE')
    result={name:np.asarray(g.glGetIntegerv(getattr(g,name))).ravel().tolist() for name in enums}
    result['GL_SAMPLE_MASK_enabled']=bool(g.glIsEnabled(g.GL_SAMPLE_MASK))
    result['sample_masks']=[int(g.glGetIntegeri_v(g.GL_SAMPLE_MASK_VALUE,i))
        for i in range(int(g.glGetIntegerv(g.GL_MAX_SAMPLE_MASK_WORDS)))]
    for name in ('GL_DEPTH_RANGE','GL_MODELVIEW_MATRIX','GL_PROJECTION_MATRIX'):
        result[name]=np.asarray(g.glGetDoublev(getattr(g,name))).tolist()
    return result


def _pixels(ctx):
    rgb,depth=ctx.read_pixels(512,512,depth=True)
    return rgb.tobytes(),depth.tobytes()


@pytest.fixture(scope='module')
def configured_env(tmp_path_factory):
    paths=[os.environ.get('CAPX_DEPTH_RUNTIME_CONFIG'),os.environ.get('CAPX_DEPTH_RUNTIME_SNAPSHOT')]
    if not all(paths):pytest.skip('opt-in configured renderer regression; requires archived config and Snapshot')
    config=json.loads(Path(paths[0]).read_text());config=config.get('config',config)
    snapshot=json.loads(Path(paths[1]).read_text())
    from capx.experiments.graph_v2 import configure_services
    configure_services(config)
    from capx.experiments.vision_tasks import make_environment
    env=make_environment(snapshot.get('task','stack'),tmp_path_factory.mktemp('depth-runtime'),snapshot['seed'],snapshot=snapshot)
    yield env
    assert env._api_calls==0
    env.close()


def test_real_rendered_frame_measures_sample_and_preserves_live_pixels(configured_env):
    from OpenGL import GL as g
    from capx.execution.public_camera import robosuite_pixel_centers
    env=configured_env;raw=env._get_observation()
    ctx=env.low_level_env.robosuite_env.sim._render_context_offscreen
    before_pixels=_pixels(ctx);before=_state(g)
    profile=env.public_depth_sampling();after=_state(g);after_pixels=_pixels(ctx)
    record=dict(profile=profile,GL_before=before,GL_after=after,
        original_rgb_bytes_unchanged=before_pixels[0]==after_pixels[0],
        original_depth_bytes_unchanged=before_pixels[1]==after_pixels[1],
        snapshot=env.reset_info,robot_API_calls=env._api_calls)
    dest=os.environ.get('CAPX_DEPTH_RUNTIME_RESULT')
    if dest:
        with Path(dest).open('x') as stream:json.dump(record,stream,indent=2)
    assert all(p['status']=='PASS' for p in profile.values()),profile
    assert before==after
    assert before_pixels==after_pixels
    corrected,metadata=robosuite_pixel_centers(raw,depth_sampling=profile,require_depth_sampling=True)
    assert metadata and corrected
    assert all(m['depth_sampling']['descriptor']['clip_depth_mode']=='GL_ZERO_TO_ONE' for m in metadata.values())
    assert env.public_depth_sampling()==profile


@pytest.mark.parametrize('enabled',[False,True])
def test_probe_failure_restores_sample_mask_enable_and_live_pixels(configured_env,monkeypatch,enabled):
    from OpenGL import GL as g
    from capx.execution.depth_sampling import _OpenGLProbe
    env=configured_env;env._get_observation()
    ctx=env.low_level_env.robosuite_env.sim._render_context_offscreen
    backend=_OpenGLProbe();description=backend.describe(ctx.con)
    original=bool(g.glIsEnabled(g.GL_SAMPLE_MASK))
    (g.glEnable if enabled else g.glDisable)(g.GL_SAMPLE_MASK)
    before_pixels=_pixels(ctx);before=_state(g)
    def fail(*a,**kw):raise RuntimeError('injected synthetic-read failure')
    try:
        with monkeypatch.context() as patch:
            patch.setattr(g,'glReadPixels',fail)
            with pytest.raises(RuntimeError,match='injected synthetic-read failure'):
                backend.sample(description)
        assert _state(g)==before
        assert _pixels(ctx)==before_pixels
    finally:(g.glEnable if original else g.glDisable)(g.GL_SAMPLE_MASK)


def test_clip_depth_mode_is_measured_and_rekeys_same_context(configured_env):
    from OpenGL import GL as g
    from capx.execution.depth_sampling import DepthSamplingProbe
    env=configured_env;env._get_observation()
    ctx=env.low_level_env.robosuite_env.sim._render_context_offscreen
    origin=int(np.asarray(g.glGetIntegerv(g.GL_CLIP_ORIGIN)).ravel()[0]);mode=int(np.asarray(g.glGetIntegerv(g.GL_CLIP_DEPTH_MODE)).ravel()[0])
    probe=DepthSamplingProbe();profiles=[]
    try:
        for clip,name in ((g.GL_NEGATIVE_ONE_TO_ONE,'GL_NEGATIVE_ONE_TO_ONE'),(g.GL_ZERO_TO_ONE,'GL_ZERO_TO_ONE')):
            g.glClipControl(origin,clip)
            before_pixels=_pixels(ctx);before=_state(g)
            profile=probe.measure(ctx)
            assert profile['status']=='PASS',profile
            assert profile['descriptor']['clip_depth_mode']==name
            assert _state(g)==before and _pixels(ctx)==before_pixels
            profiles.append(profile)
        assert profiles[0]['profile_sha256']!=profiles[1]['profile_sha256']
        assert profiles[0]['depth_offset_from_pixel_center_xy']==profiles[1]['depth_offset_from_pixel_center_xy']
    finally:g.glClipControl(origin,mode)


@pytest.mark.parametrize('group',['G0','G1'])
def test_installed_monitor_capture_and_begin_program_preserve_physics(configured_env,tmp_path,group):
    from capx.experiments.graph_v2 import install_monitor,limits
    env=configured_env;config=json.loads(Path(os.environ['CAPX_DEPTH_RUNTIME_CONFIG']).read_text())
    config=config.get('config',config)
    task=os.environ.get('CAPX_DEPTH_RUNTIME_TASK','restack')
    dest=Path(os.environ.get('CAPX_DEPTH_RUNTIME_MONITOR_OUTPUT',str(tmp_path)))/group
    dest.mkdir(parents=True,exist_ok=False)
    # This is an independent test-only physical invariance check. Private state
    # is never passed to the monitor, graph or actor.
    sim=env.low_level_env.robosuite_env.sim
    before=sim.get_state().flatten().copy();ctrl=sim.data.ctrl.copy()
    original_ik=env._apis['FrankaControlApi'].ik_solve_fn
    try:
        monitor=install_monitor(env,group,task,dest,limits(config),'none')
        monitor.capture()
        monitor.begin_program('pass',0)
        assert monitor.graph.nodes[monitor.last_observation]['valid']
        assert all(monitor.graph.nodes[i]['valid'] for i in monitor.observation_calibration_sources)
        assert np.array_equal(before,sim.get_state().flatten())
        assert np.array_equal(ctrl,sim.data.ctrl)
        assert env._api_calls==0
        with (dest/'capture-result.json').open('x') as stream:
            json.dump(dict(task=task,group=group,source_snapshot_verified=env.reset_info,
                capture='PASS',begin_program='PASS',physics_state_unchanged=True,
                controls_unchanged=True,robot_API_calls=0,llm_requests=0,
                graph=monitor.graph.snapshot()),stream,indent=2)
    finally:
        env._apis['FrankaControlApi'].ik_solve_fn=original_ik
        if hasattr(env,'_feedback_monitor'):del env._feedback_monitor
