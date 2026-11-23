import os


def test_mock_campaign_fixture_restores_runtime_environment(tmp_path,monkeypatch):
    from test_campaign import completed_development_fixture
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID','1')
    before=dict(os.environ)
    completed_development_fixture(tmp_path)
    assert dict(os.environ)==before
