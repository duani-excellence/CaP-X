"""Collect agent code and source of sandbox API tools it called (for LOC review)."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable, Mapping
from typing import Any


def _line_stats(source: str) -> dict[str, int]:
    lines = source.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    return {
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty),
    }


def collect_called_names(code: str) -> set[str]:
    """Return bare function names used in Call nodes (e.g. goto_pose(...))."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                # e.g. module.fn — rare in sandbox; keep attribute name
                names.add(func.attr)
    return names


def collect_api_sources_for_code(
    code_blocks: list[str],
    api_functions: Mapping[str, Callable[..., Any]],
) -> dict[str, Any]:
    """Match agent Call names against registered API functions and fetch sources.

    Returns a dict suitable for TrialCompleteEvent / JSON dump::

        {
          "agent_code": str,
          "agent_line_count": int,
          "agent_non_empty_line_count": int,
          "called_tools": [
            {"name": str, "source": str, "file": str|None, "line_count": int, ...}
          ],
          "tools_line_count": int,
          "tools_non_empty_line_count": int,
          "total_line_count": int,
          "total_non_empty_line_count": int,
        }
    """
    annotated = []
    for i, block in enumerate(code_blocks):
        annotated.append(f"# Code block {i}\n{block}")
    agent_code = "\n\n".join(annotated) if annotated else ""

    called: set[str] = set()
    for block in code_blocks:
        called |= collect_called_names(block)

    api_names = set(api_functions.keys())
    hit_names = sorted(called & api_names)

    tools: list[dict[str, Any]] = []
    for name in hit_names:
        fn = api_functions[name]
        source = ""
        file_path: str | None = None
        error: str | None = None
        try:
            source = inspect.getsource(fn)
            file_path = inspect.getsourcefile(fn)
        except (OSError, TypeError) as exc:
            error = f"could not get source: {exc}"
            # Bound methods sometimes need the underlying function
            try:
                underlying = inspect.unwrap(fn)
                source = inspect.getsource(underlying)
                file_path = inspect.getsourcefile(underlying)
                error = None
            except (OSError, TypeError) as exc2:
                error = f"could not get source: {exc2}"

        stats = _line_stats(source) if source else {"line_count": 0, "non_empty_line_count": 0}
        tools.append(
            {
                "name": name,
                "source": source,
                "file": file_path,
                "error": error,
                **stats,
            }
        )

    agent_stats = _line_stats(agent_code)
    tools_line_count = sum(t["line_count"] for t in tools)
    tools_non_empty = sum(t["non_empty_line_count"] for t in tools)

    return {
        "agent_code": agent_code,
        "agent_line_count": agent_stats["line_count"],
        "agent_non_empty_line_count": agent_stats["non_empty_line_count"],
        "called_tools": tools,
        "tools_line_count": tools_line_count,
        "tools_non_empty_line_count": tools_non_empty,
        "total_line_count": agent_stats["line_count"] + tools_line_count,
        "total_non_empty_line_count": agent_stats["non_empty_line_count"] + tools_non_empty,
        "unresolved_calls": sorted(called - api_names),
    }


def format_inventory_bundle(inventory: Mapping[str, Any]) -> str:
    """Single text blob: agent code then each tool source (for download / copy)."""
    parts: list[str] = [
        "# ===== Agent-generated code =====",
        inventory.get("agent_code") or "",
    ]
    for tool in inventory.get("called_tools") or []:
        parts.append("")
        parts.append(f"# ===== Tool: {tool.get('name')} =====")
        if tool.get("file"):
            parts.append(f"# file: {tool['file']}")
        if tool.get("error"):
            parts.append(f"# error: {tool['error']}")
        parts.append(tool.get("source") or "")
    parts.append("")
    parts.append(
        "# ===== LOC summary =====\n"
        f"# agent_lines={inventory.get('agent_line_count')} "
        f"(non_empty={inventory.get('agent_non_empty_line_count')})\n"
        f"# tools_lines={inventory.get('tools_line_count')} "
        f"(non_empty={inventory.get('tools_non_empty_line_count')})\n"
        f"# total_lines={inventory.get('total_line_count')} "
        f"(non_empty={inventory.get('total_non_empty_line_count')})"
    )
    return "\n".join(parts)
