"""Persistent memory: one markdown file per fact/topic under memory/.

The index (filename + first line of each memory) is injected into the system
prompt every turn, so the agent knows what it has remembered across sessions.
"""

import re

from . import config


def _path(name: str):
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower().strip()).strip("-")
    if not slug:
        raise ValueError("memory name must contain letters or digits")
    return config.MEMORY_DIR / f"{slug}.md"


def save(name: str, content: str) -> str:
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(name)
    existed = path.exists()
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return f"Memory '{path.stem}' {'updated' if existed else 'saved'}."


def read(name: str) -> str:
    path = _path(name)
    if not path.exists():
        return f"No memory named '{path.stem}'. Current index:\n{index()}"
    return path.read_text(encoding="utf-8")


def delete(name: str) -> str:
    path = _path(name)
    if not path.exists():
        return f"No memory named '{path.stem}'."
    path.unlink()
    return f"Memory '{path.stem}' deleted."


def index() -> str:
    if not config.MEMORY_DIR.exists():
        return "(no memories yet)"
    lines = []
    for f in sorted(config.MEMORY_DIR.glob("*.md")):
        first = ""
        try:
            first = f.read_text(encoding="utf-8").strip().splitlines()[0][:120]
        except (OSError, IndexError):
            pass
        lines.append(f"- {f.stem}: {first}")
    return "\n".join(lines) if lines else "(no memories yet)"
