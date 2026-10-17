import fnmatch
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

from src.core.config import get_settings
from src.core.security import resolve_safe_path
from src.tools.base import ToolDefinition

settings = get_settings()


def _ws(path: str = ".") -> Path:
    return resolve_safe_path(path, settings.workspace_path)


def _read_file(path: str) -> dict:
    p = _ws(path)
    if not p.exists():
        return {"error": f"File not found: {path}"}
    if not p.is_file():
        return {"error": f"Not a file: {path}"}
    try:
        content = p.read_text(encoding="utf-8")
        size = p.stat().st_size
        if len(content) > 5000:
            content = content[:5000] + f"\n\n... [truncated, total {size} bytes]"
        return {"path": str(p), "size": size, "lines": content.count("\n") + 1, "content": content}
    except UnicodeDecodeError:
        return {"error": f"Cannot read {path} as text (binary or unsupported encoding)"}


def _write_file(path: str, content: str) -> dict:
    p = _ws(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"status": "written", "path": str(p), "size": p.stat().st_size}


def _delete_file(path: str) -> dict:
    p = _ws(path)
    if not p.exists():
        return {"error": f"File not found: {path}"}
    p.unlink()
    return {"status": "deleted", "path": str(p)}


def _list_directory(path: str = ".") -> dict:
    p = _ws(path)
    if not p.is_dir():
        return {"error": f"Not a directory: {path}"}
    items = []
    for name in sorted(p.iterdir()):
        stat = name.stat()
        items.append({
            "name": name.name, "type": "directory" if name.is_dir() else "file",
            "size": stat.st_size if name.is_file() else None,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return {"path": str(p), "count": len(items), "items": items}


def _create_directory(path: str) -> dict:
    p = _ws(path)
    p.mkdir(parents=True, exist_ok=True)
    return {"status": "created", "path": str(p)}


def _search_files(pattern: str, directory: str = ".", recursive: bool = True) -> dict:
    p = _ws(directory)
    results = []
    iterator = p.rglob(pattern) if recursive else p.glob(pattern)
    for file_path in iterator:
        if file_path.is_file():
            results.append({
                "name": file_path.name, "path": str(file_path),
                "relative_path": str(file_path.relative_to(settings.workspace_path)),
                "size": file_path.stat().st_size,
            })
    return {"pattern": pattern, "directory": str(p), "count": len(results), "results": results}


def _search_content(pattern: str, directory: str = ".", file_pattern: str = "*", recursive: bool = True) -> dict:
    p = _ws(directory)
    regex = re.compile(pattern, re.IGNORECASE)
    results = []
    iterator = p.rglob(file_pattern) if recursive else p.glob(file_pattern)
    for file_path in iterator:
        if not file_path.is_file():
            continue
        try:
            for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if regex.search(line):
                    results.append({
                        "file": str(file_path.relative_to(settings.workspace_path)),
                        "line": line_no, "content": line.strip()[:200],
                    })
        except Exception:
            continue
    truncated = len(results) > 50
    return {"pattern": pattern, "count": len(results), "truncated": truncated, "results": results[:50]}


def create_file_tools() -> List[ToolDefinition]:
    return [
        ToolDefinition(name="read_file", description="Read the contents of a text file.",
                       parameters={"type": "object", "properties": {"path": {"type": "string", "description": "File path relative to workspace"}},
                                   "required": ["path"]}, execute=_read_file, category="file"),
        ToolDefinition(name="write_file", description="Write content to a file. Creates parent directories if needed.",
                       parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                   "required": ["path", "content"]}, execute=_write_file, category="file"),
        ToolDefinition(name="delete_file", description="Delete a file. Use with caution.",
                       parameters={"type": "object", "properties": {"path": {"type": "string"}},
                                   "required": ["path"]}, execute=_delete_file, category="file"),
        ToolDefinition(name="list_directory", description="List contents of a directory.",
                       parameters={"type": "object", "properties": {"path": {"type": "string", "default": "."}},
                                   "required": []}, execute=_list_directory, category="file"),
        ToolDefinition(name="create_directory", description="Create a new directory.",
                       parameters={"type": "object", "properties": {"path": {"type": "string"}},
                                   "required": ["path"]}, execute=_create_directory, category="file"),
        ToolDefinition(name="search_files", description="Search for files by name pattern (e.g., *.py).",
                       parameters={"type": "object", "properties": {
                           "pattern": {"type": "string"}, "directory": {"type": "string", "default": "."},
                           "recursive": {"type": "boolean", "default": True}},
                                   "required": ["pattern"]}, execute=_search_files, category="file"),
        ToolDefinition(name="search_content", description="Search file contents with regex pattern.",
                       parameters={"type": "object", "properties": {
                           "pattern": {"type": "string"}, "directory": {"type": "string", "default": "."},
                           "file_pattern": {"type": "string", "default": "*"},
                           "recursive": {"type": "boolean", "default": True}},
                                   "required": ["pattern"]}, execute=_search_content, category="file"),
    ]
