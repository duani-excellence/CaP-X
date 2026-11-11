"""Validate GPU, headless rendering and the imports used by the local pipeline."""
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

report = {
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "mujoco": mujoco.__version__,
}
assert torch.cuda.is_available(), "CUDA unavailable"
report["gpu"] = torch.cuda.get_device_name(0)
x = torch.randn(256, 256, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all()
report["cuda_matmul"] = "pass"
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
with mujoco.Renderer(m, height=128, width=128) as renderer:
    renderer.update_scene(d)
    rgb = renderer.render()
assert rgb.shape == (128, 128, 3) and np.max(rgb) > 0
report["egl_render"] = "pass"
report["imports"] = {}
for module in ["capx.envs.runner", "capx.serving.launch_pyroki_server", "capx.serving.launch_sam3_server", "capx.serving.launch_contact_graspnet_server"]:
    try:
        importlib.import_module(module)
        report["imports"][module] = "pass"
    except Exception as exc:
        report["imports"][module] = f"{type(exc).__name__}: {exc}"
path = Path("outputs/reproduction/system-check.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert all(value == "pass" for value in report["imports"].values())
