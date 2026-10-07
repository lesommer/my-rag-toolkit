from pathlib import Path

import pytest

from myrag.parse import parse_file


def test_parse_md(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text(
        "# Project Overview\n\nThe FjordHydro campaign studies internal waves.\n\n"
        "## Deployment\n\nTwelve moorings were deployed at four sites.\n\n"
        "| site | depth |\n|---|---|\n| S1 | 40 m |\n",
        encoding="utf-8",
    )
    doc = parse_file(path)
    assert doc.fmt == "md"
    assert doc.title == "notes"
    sections = [b.section for b in doc.blocks]
    assert "Project Overview" in sections
    assert "Project Overview > Deployment" in sections
    texts = [b.text for b in doc.blocks]
    assert any("Twelve moorings" in t for t in texts)
    assert any("| site | depth |" in t for t in texts)


def test_parse_txt(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("First paragraph about budgets.\n\nSecond paragraph about timelines.\n", encoding="utf-8")
    doc = parse_file(path)
    assert doc.fmt == "txt"
    assert len(doc.blocks) == 2


def test_parse_docx(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    d = docx.Document()
    d.add_heading("Compte-rendu", level=1)
    d.add_paragraph("Réunion de lancement du projet FjordHydro-2027.")
    d.add_heading("Décisions", level=2)
    d.add_paragraph("Déploiement prévu en septembre 2027.")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "jalon"
    table.cell(0, 1).text = "date"
    table.cell(1, 0).text = "déploiement"
    table.cell(1, 1).text = "2027-09"
    d.save(str(path))
    parsed = parse_file(path)
    assert parsed.fmt == "docx"
    sections = [b.section for b in parsed.blocks]
    assert "Compte-rendu" in sections
    assert "Décisions" in sections
    texts = [b.text for b in parsed.blocks]
    assert any("septembre 2027" in t for t in texts)
    assert any("| déploiement | 2027-09 |" in t for t in texts)


def test_parse_pdf_with_scan_detection(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "doc.pdf"
    pdf = pymupdf.open()
    page1 = pdf.new_page()
    page1.insert_textbox(
        pymupdf.Rect(72, 72, 500, 700),
        "Introduction\n\nThis paper studies internal wave mixing in Arctic fjords. " * 4,
        fontsize=11,
    )
    page2 = pdf.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    page2.insert_image(pymupdf.Rect(100, 100, 300, 300), pixmap=pix)
    pdf.save(str(path))
    pdf.close()

    parsed = parse_file(path)
    assert parsed.fmt == "pdf"
    assert parsed.n_pages == 2
    assert parsed.scanned_pages == [2]
    assert not parsed.is_scan
    assert any("internal wave mixing" in b.text for b in parsed.blocks)


def test_parse_pdf_fully_scanned(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "scan.pdf"
    pdf = pymupdf.open()
    for _ in range(2):
        page = pdf.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
        page.insert_image(pymupdf.Rect(100, 100, 300, 300), pixmap=pix)
    pdf.save(str(path))
    pdf.close()

    parsed = parse_file(path)
    assert parsed.is_scan
    assert parsed.scanned_pages == [1, 2]


def test_unsupported_format(tmp_path):
    path = tmp_path / "file.xyz"
    path.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_file(path)
