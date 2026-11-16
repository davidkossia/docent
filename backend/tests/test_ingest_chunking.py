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
