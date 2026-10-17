from pathlib import Path

import pytest

from src.core.exceptions import SecurityError
from src.core.security import resolve_safe_path, sanitize_input, validate_path_sandbox


class TestPathSandbox:
    @pytest.fixture
    def workspace(self, tmp_path: Path):
        ws = tmp_path / "safe_workspace"
        ws.mkdir(exist_ok=True)
        return ws

    def test_safe_path(self, workspace):
        p = resolve_safe_path("subdir/file.txt", workspace)
        assert p == workspace / "subdir" / "file.txt"

    def test_traversal_blocked(self, workspace):
        with pytest.raises(SecurityError):
            resolve_safe_path("../../../etc/passwd", workspace)

    def test_absolute_traversal_blocked(self, workspace):
        with pytest.raises(SecurityError):
            resolve_safe_path("/etc/passwd", workspace)

    def test_null_byte_removed(self):
        assert "\x00" not in sanitize_input("hello\x00world")

    def test_max_length_truncation(self):
        result = sanitize_input("x" * 20000)
        assert len(result) <= 10000

    def test_validate_returns_bool(self, workspace):
        assert validate_path_sandbox("ok.txt", workspace) is True
        assert validate_path_sandbox("../../bad", workspace) is False
