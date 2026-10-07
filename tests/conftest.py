import pytest


@pytest.fixture()
def home(tmp_path, monkeypatch):
    path = tmp_path / "rags"
    monkeypatch.setenv("MYRAG_HOME", str(path))
    return path
