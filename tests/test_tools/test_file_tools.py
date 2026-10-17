import os
from pathlib import Path

import pytest

from src.core.config import Settings
from src.core.security import resolve_safe_path
from src.tools.file_tools import _list_directory, _read_file, _search_files, _write_file


@pytest.fixture(autouse=True)
def setup_workspace(monkeypatch, tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)

    class TestSettings(Settings):
        model_config = {"env_file": ".env.test", "extra": "ignore"}
        agent_workspace_dir: str = str(ws)
        base_dir: Path = tmp_path

    monkeypatch.setattr("src.core.config.get_settings", lambda: TestSettings())
    monkeypatch.setattr("src.tools.file_tools.settings", TestSettings())
    return ws


class TestFileReadWrite:
    def test_write_and_read(self, setup_workspace):
        r = _write_file("test.txt", "Hello World")
        assert r["status"] == "written"
        r = _read_file("test.txt")
        assert "Hello World" in r["content"]

    def test_list_directory(self, setup_workspace):
        _write_file("a.txt", "a")
        _write_file("b.txt", "b")
        r = _list_directory(".")
        assert r["count"] == 2


class TestPathSecurity:
    def test_path_traversal_blocked(self, setup_workspace):
        with pytest.raises(Exception):
            resolve_safe_path("../../../etc/passwd", setup_workspace)
