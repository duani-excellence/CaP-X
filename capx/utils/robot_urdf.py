"""Resolve robot URDF paths without runtime git clones when possible."""

from __future__ import annotations

import os
from pathlib import Path


# Optional pre-installed location for deployments under /opt/capx.
_DEFAULT_BAKED_PANDA_URDF = (
    "/opt/capx/robot_descriptions/example-robot-data/robots/panda_description/urdf/panda.urdf"
)


def _robot_descriptions_roots() -> list[Path]:
    """Directories that may contain ``example-robot-data/...`` (package:// roots)."""
    roots: list[Path] = []
    for raw in (
        os.environ.get("ROBOT_DESCRIPTIONS_CACHE", ""),
        "/opt/capx/robot_descriptions",
        str(Path.home() / ".cache" / "robot_descriptions"),
    ):
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if p.is_dir() and p not in roots:
            roots.append(p)
    return roots


def resolve_robot_urdf_path(robot_urdf_or_name: str) -> str:
    """Return a filesystem URDF path, or the original name for robot_descriptions.

    Preference order for panda:
      1. Absolute/existing path argument
      2. CAPX_PANDA_URDF_PATH
      3. Baked image path under ROBOT_DESCRIPTIONS_CACHE / /opt/capx/...
      4. Fall back to name (may trigger robot_descriptions git clone)
    """
    if os.path.isfile(robot_urdf_or_name):
        return robot_urdf_or_name

    name = robot_urdf_or_name.strip()
    if name in ("panda_description", "panda"):
        env_path = os.environ.get("CAPX_PANDA_URDF_PATH", "").strip()
        if env_path and os.path.isfile(env_path):
            return env_path

        candidates = [
            Path(_DEFAULT_BAKED_PANDA_URDF),
            *[
                r / "example-robot-data/robots/panda_description/urdf/panda.urdf"
                for r in _robot_descriptions_roots()
            ],
        ]
        for path in candidates:
            if path.is_file():
                return str(path.resolve())

    return robot_urdf_or_name


def _filename_handler_for_urdf(urdf_path: Path):
    """Resolve ``package://`` and relative mesh paths for a URDF file."""
    import yourdfpy as urdfpy  # type: ignore

    urdf_dir = str(urdf_path.parent)
    package_roots = _robot_descriptions_roots()

    def handler(fname: str) -> str:
        if fname.startswith("package://"):
            rel = fname[len("package://") :]
            for root in package_roots:
                cand = root / rel
                if cand.is_file():
                    return str(cand.resolve())
        # Fall back to yourdfpy magic relative to the URDF directory.
        return urdfpy.filename_handler_magic(fname=fname, dir=urdf_dir)

    return handler


def load_urdf_for_robot(robot_urdf_or_name: str):
    """Load a yourdfpy URDF **with visual meshes** (needed for ViserUrdf)."""
    import yourdfpy as urdfpy  # type: ignore

    resolved = resolve_robot_urdf_path(robot_urdf_or_name)
    if os.path.isfile(resolved):
        path = Path(resolved)
        return urdfpy.URDF.load(
            path,
            build_scene_graph=True,
            load_meshes=True,
            filename_handler=_filename_handler_for_urdf(path),
        )

    from robot_descriptions.loaders.yourdfpy import load_robot_description  # type: ignore

    return load_robot_description(resolved)
