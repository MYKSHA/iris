"""Tool definitions and implementations, shared by every provider.

Each tool is described once with a JSON schema; providers translate that to
their own wire format. `confirm_callback` is set by interactive interfaces
(the CLI) to approve shell commands; non-interactive interfaces (Telegram)
leave it None, in which case shell commands require IRIS_YOLO=1.
"""

import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

from . import config, memory

MAX_OUTPUT = 12_000
confirm_callback: Optional[Callable[[str], bool]] = None


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n... [truncated {len(text) - MAX_OUTPUT} chars]"


def run_command(command: str) -> str:
    approved = config.YOLO
    if not approved and confirm_callback is not None:
        approved = confirm_callback(command)
    if not approved:
        return (
            "Command NOT executed: the user declined, or shell approval is not "
            "available in this interface (IRIS_YOLO=1 enables it). Tell the user "
            "what you wanted to run and why."
        )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120 seconds."
    out = proc.stdout or ""
    if proc.stderr:
        out += "\n[stderr]\n" + proc.stderr
    out = out.strip()
    return _truncate(out if out else f"(no output, exit code {proc.returncode})")


def read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: {p} does not exist."
    if p.is_dir():
        return list_dir(path)
    try:
        return _truncate(p.read_text(encoding="utf-8", errors="replace"))
    except OSError as e:
        return f"Error reading {p}: {e}"


def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {p}."
    except OSError as e:
        return f"Error writing {p}: {e}"


def list_dir(path: str) -> str:
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"Error: {p} is not a directory."
    entries = []
    for child in sorted(p.iterdir()):
        suffix = "/" if child.is_dir() else f"  ({child.stat().st_size} bytes)"
        entries.append(child.name + suffix)
    return _truncate("\n".join(entries) if entries else "(empty directory)")


def web_search(query: str, max_results: int = 5) -> str:
    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=int(max_results)))
    except Exception as e:  # network errors, rate limits, parser changes
        return f"Web search failed: {e}"
    if not results:
        return "No results found."
    lines = [
        f"- {r.get('title', '?')}\n  {r.get('href', '')}\n  {r.get('body', '')}"
        for r in results
    ]
    return _truncate("\n".join(lines))


# ── Registry ─────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_command",
        "description": (
            "Run a PowerShell command on the user's Windows machine and return its "
            "output. Use for anything the other tools don't cover: git, package "
            "managers, system info, launching apps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The PowerShell command to run."}
            },
            "required": ["command"],
        },
        "func": run_command,
    },
    {
        "name": "read_file",
        "description": "Read a text file from disk.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or ~-relative path."}},
            "required": ["path"],
        },
        "func": read_file,
    },
    {
        "name": "write_file",
        "description": "Write text content to a file, creating parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        "func": write_file,
    },
    {
        "name": "list_dir",
        "description": "List the entries of a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "func": list_dir,
    },
    {
        "name": "web_search",
        "description": (
            "Search the web (DuckDuckGo) and return titles, URLs, and snippets. "
            "Use when the answer depends on current information you don't have."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "1-10, default 5."},
            },
            "required": ["query"],
        },
        "func": web_search,
    },
    {
        "name": "memory_save",
        "description": (
            "Save a lasting fact, preference, or lesson to persistent memory so future "
            "sessions know it. Call this when the user shares something durable about "
            "themselves or their projects, or explicitly asks you to remember something. "
            "Use a short kebab-case name; saving to an existing name overwrites it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short kebab-case identifier."},
                "content": {
                    "type": "string",
                    "description": "The fact to remember. First line should summarize it.",
                },
            },
            "required": ["name", "content"],
        },
        "func": memory.save,
    },
    {
        "name": "memory_read",
        "description": "Read the full content of one memory from the index.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        "func": memory.read,
    },
    {
        "name": "memory_delete",
        "description": "Delete a memory that is wrong or no longer relevant.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        "func": memory.delete,
    },
]

_BY_NAME = {t["name"]: t for t in TOOLS}


def execute(name: str, args: dict) -> str:
    """Run a tool by name; always return a string (errors included) for the model."""
    tool = _BY_NAME.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available: {', '.join(_BY_NAME)}"
    try:
        return str(tool["func"](**args))
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}. Expected schema: {json.dumps(tool['input_schema'])}"
    except Exception as e:
        return f"Error while running {name}: {e}"
