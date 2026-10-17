import os
import re
from pathlib import Path

from src.core.exceptions import SecurityError


# Maximum user input length to prevent abuse
MAX_INPUT_LENGTH = 10_000

# Allowed characters in file paths (prevent path injection)
_SAFE_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9._\-/\\ ]+$")


def resolve_safe_path(path: str, workspace_dir: Path) -> Path:
    """
    Resolve a path safely within the workspace directory.
    Raises SecurityError if the resolved path escapes the workspace.
    """
    workspace_dir = workspace_dir.resolve()

    # Reject paths with suspicious characters
    if not _SAFE_PATH_PATTERN.match(path):
        raise SecurityError(
            f"Path contains invalid characters: {path}",
            detail="Only alphanumeric, dots, dashes, underscores, slashes, and spaces are allowed.",
        )

    p = Path(path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (workspace_dir / p).resolve()

    # Ensure the resolved path is within workspace
    try:
        resolved.relative_to(workspace_dir)
    except ValueError:
        raise SecurityError(
            f"Path traversal detected: '{path}' resolves outside workspace",
            detail=f"Resolved path: {resolved}",
        )

    return resolved


def validate_path_sandbox(path: str, workspace_dir: Path) -> bool:
    """Check if a path is safely within the workspace. Returns True if safe."""
    try:
        resolve_safe_path(path, workspace_dir)
        return True
    except SecurityError:
        return False


def sanitize_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """
    Sanitize user input.
    - Truncate to max_length
    - Strip null bytes
    """
    if not isinstance(text, str):
        raise SecurityError("Input must be a string")

    text = text.replace("\x00", "")

    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()
