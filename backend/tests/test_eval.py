import json
from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from docent import evalrun
from docent.evalrun import (
    add_baseline_comparison,
    bootstrap_ci,
    is_evidence_hit,
    jaccard,
    mrr_at,
    normalize,
    recall_at,
    select_setup,
    split_metrics,
    validate_questions,
    won_lost,
    write_report,
)
from docent.ingest import Section

# ---------- evidence hit ----------

EV = "the quick brown fox jumps over the lazy dog"  # 43 characters


def test_hit_at_exactly_70_percent():
    ev = "0123456789"  # 10 chars; 7 shared is the edge
    assert is_evidence_hit("xx 0123456 yy", ev)
    assert not is_evidence_hit("xx 012345 yy", ev)  # 6 of 10


def test_hit_whole_evidence_inside_longer_chunk():
    assert is_evidence_hit("intro text. " + EV + ". more text", EV)


def test_hit_normalizes_case_whitespace_and_hyphenation():
    chunk = "The quick  brown fox\njumps over the la-\nzy dog"
    assert is_evidence_hit(chunk, EV)
    assert normalize("exam-\nple   TEXT\n\nhere") == "example text here"


def test_no_hit_for_unrelated_text():
    assert not is_evidence_hit("completely different words about databases", EV)


# ---------- metrics ----------

def test_metrics_hand_computed():
    ranks = [1, 3, None, 5, 2, 7]  # rank 7 is outside the top 5
    assert recall_at(ranks, 1) == pytest.approx(1 / 6)
    assert recall_at(ranks, 3) == pytest.approx(3 / 6)
    assert recall_at(ranks, 5) == pytest.approx(4 / 6)
    assert mrr_at(ranks, 5) == pytest.approx((1 + 1 / 3 + 1 / 5 + 1 / 2) / 6)


def test_split_metrics_columns():
    rows = [{"rank": 1, "section_rank": 1}, {"rank": None, "section_rank": 2}]
    m = split_metrics(rows)
    assert m == {"n": 2, "recall@1": 0.5, "recall@3": 0.5, "recall@5": 0.5, "mrr@5": 0.5, "section_recall@5": 1.0}


def test_bootstrap_is_deterministic_and_bounds_the_mean():
    ranks = [1] * 30 + [None] * 30
    a, b = bootstrap_ci(ranks), bootstrap_ci(ranks)
    assert a == b
    assert a[0] < 0.5 < a[1]
    assert bootstrap_ci([1] * 10) == [1.0, 1.0]


def test_won_lost_counts():
    mine = [1, None, 2, None]
    base = [None, 1, 3, None]
    assert won_lost(mine, base) == {"won": 1, "lost": 1}


def _row(strategy, embedder, dev5, chunks, ranks):
    qs = [{"id": f"q{i}", "split": "test", "type": "lookup", "rank": r, "section_rank": r} for i, r in enumerate(ranks)]
    return {"strategy": strategy, "embedder": embedder, "chunks": chunks,
            "metrics": {"dev": {"recall@5": dev5, "mrr@5": 0.5}, "test": {}}, "questions": qs}


def test_baseline_comparison_and_selection_rule():
    base = _row("fixed-1000", "e", 0.5, 100, [1, None, None])
    a = _row("section-800", "e", 0.7, 300, [1, 2, None])
    b = _row("section-500", "e", 0.7, 200, [None, 2, 1])  # ties a on dev; fewer chunks wins
    rows = [base, a, b]
    add_baseline_comparison(rows)
    assert base["vs_baseline"] is None
    assert a["vs_baseline"] == {"won": 1, "lost": 0}
    assert b["vs_baseline"] == {"won": 2, "lost": 1}
    assert select_setup(rows) is b


# ---------- validator ----------

TOY_TEXT = (
    "Python is an interpreted language: the source code is translated one line at a time while it runs.\n"
    "Loops repeat statements until a condition changes, which lets programs process many items.\n"
)
SECTIONS = [
    Section("py-1.1", "py", "Toy", "1.1 Basics", 1, [(1, TOY_TEXT)]),
    Section("cs-1.1", "cs", "Toy", "1.1 Intro", 1, [(1, "Networks move packets between machines across links.\n")]),
]
GOOD = {
    "id": "q001", "split": "dev", "type": "lookup",
    "question": "How does Python translate source code while a program is running on a computer?",
    "answer": "One line at a time.", "gold_section": "py-1.1",
    "evidence": "Python is an interpreted language: the source code is translated one line at a time while it runs.",
    "must_not_appear": None,
}
UNANS = {**GOOD, "id": "q002", "type": "unanswerable-offtopic", "answer": None, "gold_section": None,
         "evidence": None, "must_not_appear": ["quantum annealing"],
         "question": "How does quantum annealing hardware compare with gate based machines for optimization?"}


def _validate(tmp_path, rows):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return validate_questions(p, SECTIONS, check_counts=False)


def test_valid_rows_pass(tmp_path):
    assert _validate(tmp_path, [GOOD, UNANS]) == []


def test_rule_missing_field(tmp_path):
    bad = {k: v for k, v in GOOD.items() if k != "evidence"}
    assert any("missing fields" in e for e in _validate(tmp_path, [bad]))


def test_rule_question_length(tmp_path):
    assert any("8 to 30 words" in e for e in _validate(tmp_path, [{**GOOD, "question": "Too short?"}]))


def test_rule_evidence_must_be_verbatim(tmp_path):
    bad = {**GOOD, "evidence": "Python is a compiled language: the source code is translated all at once, always."}
    assert any("verbatim" in e for e in _validate(tmp_path, [bad]))


def test_rule_evidence_length(tmp_path):
    assert any("40 to 300" in e for e in _validate(tmp_path, [{**GOOD, "evidence": "Python is an interpreted"}]))


def test_rule_paraphrase_overlap(tmp_path):
    bad = {**GOOD, "type": "paraphrase",
           "question": "Python interpreted language source code translated one line at a time while running?"}
    assert any("paraphrase overlap" in e for e in _validate(tmp_path, [bad]))
    assert jaccard("alpha beta", "alpha beta") == 1.0


def test_rule_must_not_appear_absent_from_corpus(tmp_path):
    bad = {**UNANS, "must_not_appear": ["interpreted language"]}
    assert any("occurs in the corpus" in e for e in _validate(tmp_path, [bad]))


def test_rule_at_most_two_per_section(tmp_path):
    rows = [{**GOOD, "id": f"q00{i}"} for i in range(1, 4)]
    assert any("limit is 2" in e for e in _validate(tmp_path, rows))


def test_rule_unknown_section_and_duplicate_id_and_code_book(tmp_path):
    errs = _validate(tmp_path, [{**GOOD, "gold_section": "py-9.9"}, GOOD, GOOD])
    assert any("not a section" in e for e in errs)
    assert any("duplicate id" in e for e in errs)
    code_cs = {**GOOD, "id": "q009", "type": "code", "gold_section": "cs-1.1",
               "evidence": "Networks move packets between machines across links, said the toy book text."}
    assert any("Python-book" in e for e in _validate(tmp_path, [code_cs]))


def test_rule_counts(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps(GOOD) + "\n")
    errs = validate_questions(p, SECTIONS, check_counts=True)
    assert any("expected 144" in e for e in errs)
    assert any("dev/paraphrase is 0, expected 25" in e for e in errs)


# ---------- report ----------

def test_report_rewrites_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- eval:start -->\nold\n<!-- eval:end -->\nafter\n")
    results = tmp_path / "results"
    results.mkdir()
    row = _row("fixed-1000", "e", 0.5, 100, [1, None])
    row["metrics"] = {sp: {"recall@1": 0.5, "recall@3": 0.5, "recall@5": 0.5, "mrr@5": 0.5,
                           "section_recall@5": 0.5, "n": 2} for sp in ("dev", "test")}
    row["test_recall@5_ci95"] = [0.0, 1.0]
    (results / "retrieval.json").write_text(json.dumps([row]))
    write_report(readme, results)
    text = readme.read_text()
    assert text.startswith("before\n<!-- eval:start -->\n") and text.endswith("<!-- eval:end -->\nafter\n")
    assert "old" not in text and "fixed-1000" in text


def test_report_requires_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n")
    with pytest.raises(SystemExit):
        write_report(readme, tmp_path)


# ---------- retrieval with a fake index ----------

@dataclass
class _Hit:
    doc: Document


def test_evidence_rank_and_section_rank():
    hits = [_Hit(Document(page_content="unrelated", metadata={"section_ids": ["a"]})),
            _Hit(Document(page_content="xx " + EV, metadata={"section_ids": ["b"]}))]
    assert evalrun.evidence_rank(hits, EV) == 2
    assert evalrun.section_rank(hits, "b") == 2
    assert evalrun.section_rank(hits, "z") is None
