"""Regression check: omitted Web output limit inherits CLI defaults."""
import threading
from unittest.mock import patch

from fastapi.testclient import TestClient
from capx.web.server import create_app

seen = []
ready = threading.Event()

async def capture_trial(*, session, args):
    seen.append(args.max_tokens)
    ready.set()

app = create_app()
app.state.default_max_tokens = 4096
payload = {
    "config_path": "env_configs/cube_stack/franka_robosuite_cube_stack_privileged.yaml",
    "model": "qwen2.5-coder:7b",
    "server_url": "http://127.0.0.1:8000/v1/chat/completions",
}
with patch("capx.web.server.run_trial_async", capture_trial), patch(
    "capx.web.server._load_config", return_value=({}, {}, [])
), TestClient(app) as client:
    assert client.get("/api/default-config").json()["max_tokens"] == 4096
    for supplied, expected in [(None, 4096), (128, 128)]:
        ready.clear()
        body = {**payload}
        if supplied is not None:
            body["max_tokens"] = supplied
        response = client.post("/api/start-trial", json=body)
        assert response.status_code == 200, response.text
        assert ready.wait(5), "Trial did not start"
        assert seen[-1] == expected, seen
    assert client.post("/api/start-trial", json={**payload, "max_tokens": 0}).status_code == 422
print("PASS: Web inherits CLI limit, respects request override, rejects zero.")
