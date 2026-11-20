import pytest

from docent.chunking import STRATEGIES, build_chunks
from docent.corpus import BOOKS
from docent.ingest import Section, clean_text, load_sections

needs_corpus = pytest.mark.skipif(
    not all(b.path.exists() for b in BOOKS), reason="run `python -m docent download` first"
)


def test_clean_text_drops_footers_and_hyphenation():
    raw = "Intro text with exam-\nple word\n2.5 • Dividing integers     51\nAccess for free at openstax.org\n52     2 • Expressions\nend"
    out = clean_text(raw)
    assert "example" in out
    assert "openstax.org" not in out
    assert "Dividing integers" not in out
    assert "2 • Expressions" not in out
    assert out.endswith("end")


def _toy_sections():
    a = Section("t-1.1", "t", "Toy Book", "1.1 Alpha", 1, [(1, "alpha line\n" * 25), (2, "alpha line\n" * 25)])
    b = Section("t-1.2", "t", "Toy Book", "1.2 Beta", 1, [(2, "beta line\n" * 25), (3, "beta line\n" * 25)])
    return [a, b]


def test_section_aware_chunks_never_cross_sections():
    docs = build_chunks(STRATEGIES["section-800"], _toy_sections())
    assert docs
    for d in docs:
        assert d.metadata["section_ids"] == [d.metadata["section_id"]]
        assert d.page_content.startswith("Toy Book | ")


def test_fixed_chunks_can_cross_sections_and_record_it():
    docs = build_chunks(STRATEGIES["fixed-1000"], _toy_sections())
    assert any(len(d.metadata["section_ids"]) > 1 for d in docs)
    assert all(d.metadata["page_start"] <= d.metadata["page_end"] for d in docs)


@needs_corpus
def test_corpus_has_500_plus_pages():
    sections = load_sections()
    pages = {(s.book, p) for s in sections for p, _ in s.pages}
    assert len(pages) >= 500
    assert len(sections) > 100


def test_ladder_matches_spec_and_drops_fixed_400():
    assert list(STRATEGIES) == [
        "fixed-1000", "fixed-800", "clean-800", "section-800-nohdr", "section-800", "section-500", "section-1200",
    ]
    spec = {
        "fixed-1000": (1000, 200, False, False, False),
        "fixed-800": (800, 100, False, False, False),
        "clean-800": (800, 100, True, False, False),
        "section-800-nohdr": (800, 100, True, True, False),
        "section-800": (800, 100, True, True, True),
        "section-500": (500, 80, True, True, True),
        "section-1200": (1200, 150, True, True, True),
    }
    for name, (size, ov, clean, by_sec, hdr) in spec.items():
        s = STRATEGIES[name]
        assert (s.chunk_size, s.overlap, s.clean, s.by_section, s.header) == (size, ov, clean, by_sec, hdr)


def test_nohdr_has_no_header_and_keeps_metadata_keys():
    docs = build_chunks(STRATEGIES["section-800-nohdr"], _toy_sections())
    assert docs and not any(d.page_content.startswith("Toy Book | ") for d in docs)
    for key in ("book", "book_title", "section_id", "section_title", "section_ids", "page_start", "page_end", "chunk_id"):
        assert key in docs[0].metadata
    assert docs[0].metadata["chunk_id"].startswith("section-800-nohdr:")


def test_clean_strategy_strips_footers_but_fixed_keeps_them():
    sec = Section("t-1.1", "t", "Toy Book", "1.1 Alpha", 1,
                  [(1, "body text here\nAccess for free at openstax.org\nmore body text")])
    clean = build_chunks(STRATEGIES["clean-800"], [sec])
    raw = build_chunks(STRATEGIES["fixed-800"], [sec])
    assert all("openstax.org" not in d.page_content for d in clean)
    assert any("openstax.org" in d.page_content for d in raw)
