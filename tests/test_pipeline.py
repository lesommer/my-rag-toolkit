import os

import pytest

from myrag import pipeline


def test_create_add_discover(home):
    project = pipeline.create_project(home, "demo", "a test corpus")
    assert (project / "config.yaml").exists()
    assert (project / "docs").is_dir()
    assert (project / "index").is_dir()
    assert pipeline.discover_projects(home) == ["demo"]

    with pytest.raises(FileExistsError):
        pipeline.create_project(home, "demo")


def test_add_documents_filters_unsupported(home, tmp_path):
    pipeline.create_project(home, "demo")
    src = tmp_path / "src"
    src.mkdir()
    (src / "note.md").write_text("# Title\n\nBody text.", encoding="utf-8")
    (src / "binary.xyz").write_text("junk", encoding="utf-8")
    copied = pipeline.add_documents(home, "demo", [str(src)])
    assert [p.name for p in copied] == ["note.md"]

    with pytest.raises(ValueError):
        pipeline.add_documents(home, "demo", [str(src / "binary.xyz")])


@pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="downloads bge-m3 (~2.3 GB)")
def test_build_and_query_real(home, tmp_path):
    pipeline.create_project(home, "real", "live model test")
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text(
        "# Deployment\n\nTwelve moorings were deployed at four sites in the fjord.\n" * 20,
        encoding="utf-8",
    )
    pipeline.add_documents(home, "real", [str(src / "a.md")])
    report = pipeline.build_project(home, "real", reset=True, log=lambda m: None)
    assert report.ok_files == 1
    assert report.total_chunks > 0

    result = pipeline.query_project(home, "real", "How many moorings were deployed?", use_llm=False)
    assert result.contexts
    assert "moorings" in result.contexts[0].text.lower() or "sites" in result.contexts[0].text.lower()
