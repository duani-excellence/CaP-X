"""Shared Viser port registry for web UI reverse-proxy."""

from __future__ import annotations

from urllib.request import urlopen

_VISER_PORTS = list(range(8080, 8090))
_viser_port_cache: int | None = None
_active_viser_port: int | None = None


def invalidate_viser_port_cache() -> None:
    """Drop cached Viser port (call when stopping/replacing a trial env)."""
    global _viser_port_cache, _active_viser_port
    _viser_port_cache = None
    _active_viser_port = None


def set_active_viser_port(port: int | None) -> None:
    """Remember the Viser port owned by the current trial env."""
    global _viser_port_cache, _active_viser_port
    _active_viser_port = port
    _viser_port_cache = port


def find_viser_port() -> int | None:
    """Probe candidate ports; prefer the active trial's registered port."""
    global _viser_port_cache

    preferred = _active_viser_port if _active_viser_port is not None else _viser_port_cache
    if preferred is not None:
        try:
            urlopen(f"http://localhost:{preferred}/", timeout=0.5)
            _viser_port_cache = preferred
            return preferred
        except Exception:
            _viser_port_cache = None

    for port in _VISER_PORTS:
        try:
            urlopen(f"http://localhost:{port}/", timeout=0.5)
            _viser_port_cache = port
            return port
        except Exception:
            continue
    return None
