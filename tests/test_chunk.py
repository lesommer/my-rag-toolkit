from myrag.chunk import (
    add_overlap,
    chunk_document,
    chunk_text_block,
    count_tokens,
    has_repeating_loop,
    is_affiliation_fragment,
    is_boilerplate,
    is_digit_heavy,
    is_reference_block,
    is_url_only,
    split_sentences,
)
from myrag.parse import ParsedBlock, ParsedDocument


def test_count_tokens():
    assert count_tokens("hello world") > 0
    assert count_tokens("one two three") > count_tokens("one two")


def test_sentence_split():
    text = "Ceci est une phrase. Here is another one! Et une troisième ?"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_chunk_text_block_respects_budget():
    text = " ".join(f"Sentence number {i} explains something about ocean mixing." for i in range(200))
    chunks = chunk_text_block(text, max_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert count_tokens(c) <= 150


def test_chunk_text_block_single_when_small():
    text = "Short text."
    assert chunk_text_block(text, max_tokens=100) == [text]


def test_add_overlap_shares_content():
    text = " ".join(f"Sentence {i} talks about fjord hydrography with detail {i}." for i in range(60))
    base = chunk_text_block(text, max_tokens=60)
    overlapped = add_overlap(base, 0.5, 60)
    assert len(overlapped) == len(base)
    last_sent = split_sentences(base[0])[-1]
    assert last_sent in overlapped[1]


def test_is_url_only():
    assert is_url_only("https://example.com/page")
    assert not is_url_only("see https://example.com for more")


def test_is_reference_block():
    refs = "\n".join(
        f"Author{chr(65 + i)}, J., AuthorB, K., and AuthorC, L. (20{10 + i}). Title of paper {i}. Journal of Things, 1(2), 10.1234/abcd.{i}"
        for i in range(5)
    )
    assert is_reference_block(refs)
    assert not is_reference_block(
        "This paragraph discusses ocean mixing. It mentions one DOI: 10.1234/abcd.1 and one year (2020)."
    )


def test_is_digit_heavy_skips_tables():
    table = "| a | b |\n|---|---|\n| 1.2 | 3.4 |\n| 5.6 | 7.8 |"
    assert not is_digit_heavy(table)
    garbage = "12 34 56 78 90 12 34 56 78 90 12 34 56 78 90 ° ° ° ° ± ± ± ± = = = ="
    assert is_digit_heavy(garbage)


def test_is_boilerplate():
    text = "Copyright and moral rights for this publication are retained. Take down policy applies."
    assert is_boilerplate(text)


def test_is_affiliation_fragment():
    aff = "Department of Oceanography, University of Bergen, Norway. Institute of Marine Research, Bergen."
    assert is_affiliation_fragment(aff)


def test_has_repeating_loop():
    unit = "Some repeated visual garbage line. "
    assert has_repeating_loop(unit * 10)
    assert not has_repeating_loop("A perfectly normal paragraph with varied sentences. " * 2)


def _make_doc(fmt="md", blocks=None):
    if blocks is None:
        blocks = [
            ParsedBlock(text="# Heading", section="Heading"),
            ParsedBlock(
                text=" ".join(
                    f"Sentence {i} explains aspect {i} of fjord mixing with measurements." for i in range(30)
                ),
                section="Heading",
            ),
        ]
    return ParsedDocument(
        doc_id="test.md",
        source_path="/tmp/test.md",
        title="Test Doc",
        fmt=fmt,
        blocks=blocks,
    )


def test_chunk_document_metadata():
    doc = _make_doc()
    chunks = chunk_document(doc)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.doc_id == "test.md"
    assert c.title == "Test Doc"
    assert c.section == "Heading"
    assert c.page == -1
    assert c.text_with_prefix.startswith("Doc: Test Doc\nSection: Heading\n---\n")


def test_chunk_document_dedup():
    para = "Unique paragraph content about internal waves. " * 20
    doc = _make_doc(
        blocks=[
            ParsedBlock(text=para, section="A"),
            ParsedBlock(text=para, section="B"),
        ]
    )
    chunks = chunk_document(doc)
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))


def test_chunk_document_pdf_page_tags():
    blocks = [
        ParsedBlock(
            text="Page one content. "
            + " ".join(f"Detail {i} concerns mooring depth at site {i % 4}." for i in range(25)),
            page=1,
        ),
        ParsedBlock(
            text="Page two content. "
            + " ".join(f"Budget line {i} covers instrument cost {i} thousand euros." for i in range(25)),
            page=2,
        ),
    ]
    doc = ParsedDocument(
        doc_id="t.pdf", source_path="/tmp/t.pdf", title="T", fmt="pdf", blocks=blocks, n_pages=2
    )
    chunks = chunk_document(doc)
    pages = {c.page for c in chunks}
    assert pages == {1, 2}
