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


def _write_structured_pdf(path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 400, 500, 700),
        "The FjordHydro campaign studies internal wave mixing in the fjord. " * 3,
        fontsize=11,
    )
    cols = [72, 172, 272, 372]
    rows = [72, 122, 172, 222]
    for x in cols:
        page.draw_line((x, rows[0]), (x, rows[-1]))
    for y in rows:
        page.draw_line((cols[0], y), (cols[-1], y))
    cells = [["site", "depth", "instrument"], ["S1", "40 m", "CTD"], ["S2", "320 m", "mooring"]]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            page.insert_text((cols[c] + 4, rows[r] + 15), val, fontsize=10)
    page.insert_text((72, 300), "Project website", fontsize=11)
    page.insert_link(
        {
            "kind": pymupdf.LINK_URI,
            "from": pymupdf.Rect(72, 288, 200, 305),
            "uri": "https://fjordhydro.example.org",
        }
    )
    pdf.save(str(path))
    pdf.close()


def test_parse_pdf_tables_and_links(tmp_path):
    path = tmp_path / "structured.pdf"
    _write_structured_pdf(path)
    parsed = parse_file(path)
    table_blocks = [b for b in parsed.blocks if b.text.startswith("| ")]
    assert len(table_blocks) == 1
    assert "| site | depth | instrument |" in table_blocks[0].text
    assert "| S1 | 40 m | CTD |" in table_blocks[0].text
    assert table_blocks[0].page == 1
    stray = [b for b in parsed.blocks if "S1" in b.text and not b.text.startswith("| ")]
    assert not stray
    url_blocks = [b for b in parsed.blocks if b.kind == "urls"]
    assert len(url_blocks) == 1
    assert "URLs in this document:" in url_blocks[0].text
    assert "- p.1: https://fjordhydro.example.org [Project website]" in url_blocks[0].text
    assert any("internal wave mixing" in b.text for b in parsed.blocks if b.kind == "text")


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
