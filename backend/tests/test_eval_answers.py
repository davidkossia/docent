import json
from types import SimpleNamespace as NS

import pytest
from langchain_core.documents import Document

from docent import evalrun
from docent.evalrun import (
    Question,
    answer_metrics,
    answers_table,
    estimate_usd,
    evaluate_question,
    judge_answer,
    parse_setup,
    run_answers,
)
from docent.index import Hit

EVIDENCE = "a list is an ordered collection of items that you can change after it is created"
OFFTOPIC = "unanswerable-offtopic"


def _q(id, type="lookup", split="test", answer="Lists are mutable.", evidence=EVIDENCE):
    un = type.startswith("unanswerable")
    return Question(id, split, type, f"Question {id}?", None if un else answer, None if un else "py-1.1",
                    None if un else evidence, ["xyz"] if un else None)


class Retriever:
    def search(self, question, k):
        meta = {"chunk_id": "c1", "section_id": "py-1.1", "section_title": "1.1 Lists", "book_title": "Py",
                "page_start": 1, "page_end": 1}
        return [Hit(Document(page_content="Intro. " + EVIDENCE + ". More.", metadata=meta), 0.9),
                Hit(Document(page_content="Unrelated text about files.", metadata={**meta, "chunk_id": "c2"}), 0.5)]


class FakeClient:
    """Answers by question id: ids in `refuse` get the no-answer text, ids in `cite_wrong` cite passage 2."""

    def __init__(self, refuse=(), cite_wrong=(), judge=True, tokens=(1000, 500), fail=()):
        self.refuse, self.cite_wrong, self.judge, self.tokens, self.fail = set(refuse), set(cite_wrong), judge, tokens, set(fail)
        self.calls = []
        outer = self

        class Messages:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                usage = NS(input_tokens=outer.tokens[0], output_tokens=outer.tokens[1])
                if "format" in kw["output_config"]:  # the judge call
                    return NS(stop_reason="end_turn", model="m", usage=usage,
                              content=[NS(type="text", text=json.dumps({"correct": outer.judge}))])
                qid = kw["messages"][0]["content"][-1]["text"].split()[1].rstrip("?")
                if qid in outer.fail:
                    raise RuntimeError("boom")
                if qid in outer.refuse:
                    blocks = [NS(type="text", text="The documentation does not answer this question.", citations=None)]
                else:
                    idx = 1 if qid in outer.cite_wrong else 0
                    blocks = [NS(type="text", text="Lists are mutable.",
                                 citations=[NS(document_index=idx, cited_text="x")])]
                return NS(stop_reason="end_turn", model="m", usage=usage, content=blocks)

        self.beta = NS(messages=Messages)


def _write_qs(path, qs):
    path.write_text("\n".join(json.dumps(q.__dict__) for q in qs) + "\n")


def _body(status, citations=(), passages=("Intro. " + EVIDENCE,)):
    return {"status": status, "segments": [{"text": "t", "citations": [{"passage": p, "cited_text": "x"} for p in citations]}],
            "passages": [{"text": t} for t in passages]}


# ---------- each metric on a hand-built case ----------

def test_grounded_needs_a_cited_chunk_with_the_evidence():
    q = _q("a")
    assert evaluate_question(q, _body("answered", [1]))["grounded"]
    assert not evaluate_question(q, _body("answered", [2], passages=("Intro. " + EVIDENCE, "other text")))["grounded"]
    assert not evaluate_question(q, _body("answered", []))["grounded"]
    assert not evaluate_question(q, _body("no_answer", [1]))["grounded"]


def test_metrics_on_hand_built_rows():
    def row(id, answerable, status, grounded=False, correct=None):
        return {"id": id, "answerable": answerable, "status": status, "grounded": grounded, "correct": correct,
                "cost_usd": 0.5}

    rows = [
        row("a", True, "answered", True, True),
        row("b", True, "answered", True, False),
        row("c", True, "answered", False, True),
        row("d", True, "no_answer"),
        row("e", False, "no_answer"),
        row("f", False, "answered"),
        row("g", False, "no_answer"),
        row("h", False, "error"),
    ]
    m = answer_metrics(rows)
    assert m["grounded_rate"] == 0.5  # 2 of 4 answerable
    assert m["judged_correct_rate"] == 0.5  # 2 of 4
    assert m["refusal_accuracy"] == 0.5  # 2 of 4 unanswerable; the error is a miss
    assert m["wrong_refusal_rate"] == 0.25  # 1 of 4
    assert m["errors"] == 1 and m["cost_usd"] == 4.0


def test_metrics_with_no_rows_are_none():
    assert answer_metrics([])["grounded_rate"] is None


def test_judge_sees_only_question_reference_and_answer():
    c = FakeClient(judge=False)
    assert judge_answer(c, "Why?", "Because.", "No idea.") is False
    call = c.calls[0]
    assert call["output_config"]["effort"] == "low" and call["output_config"]["format"]["type"] == "json_schema"
    prompt = call["messages"][0]["content"]
    assert "Why?" in prompt and "Because." in prompt and "No idea." in prompt
    assert "Passage" not in prompt and "citation" not in prompt.lower()


def test_judge_refusal_is_none():
    c = NS(beta=NS(messages=NS(create=lambda **kw: NS(stop_reason="refusal", content=[], usage=NS(input_tokens=1, output_tokens=1)))))
    assert judge_answer(c, "q", "r", "a") is None


def test_parse_setup():
    assert parse_setup("section-800__local-bge-small") == ("section-800", "local-bge-small")
    with pytest.raises(SystemExit):
        parse_setup("section-800")
    with pytest.raises(SystemExit):
        parse_setup("nope__local-bge-small")


# ---------- the run ----------

QS = [_q("q1"), _q("q2"), _q("q3", OFFTOPIC), _q("q4", split="dev")]
SETUP = "section-800__local-bge-small"


def _run(tmp_path, client, max_usd=10.0, yes=True, setups=(SETUP,), qs=QS):
    qp = tmp_path / "q.jsonl"
    _write_qs(qp, qs)
    return run_answers(list(setups), max_usd, yes, qp, tmp_path / "answers.json", client=client,
                       retriever_factory=lambda s: Retriever())


def test_run_scores_the_test_split_only(tmp_path):
    c = FakeClient(refuse=["q3"])
    data = _run(tmp_path, c)
    rows = data["setups"][SETUP]["rows"]
    assert [r["id"] for r in rows] == ["q1", "q2", "q3"]  # the dev question is skipped
    m = data["setups"][SETUP]["metrics"]
    assert (m["grounded_rate"], m["judged_correct_rate"], m["refusal_accuracy"], m["wrong_refusal_rate"]) == (1.0, 1.0, 1.0, 0.0)
    assert len(c.calls) == 5  # 3 answer calls + 2 judge calls (no judge for the unanswerable)
    assert (tmp_path / "answers.json").exists()


def test_cost_comes_from_usage(tmp_path):
    data = _run(tmp_path, FakeClient(tokens=(1_000_000, 100_000)), max_usd=1000)
    per_call = 4 + 2  # $4 input + $2 output per call
    assert data["cost_usd"] == pytest.approx(per_call * 5)


def test_dry_run_prints_counts_and_estimate_and_makes_no_calls(tmp_path, capsys):
    c = FakeClient()
    assert _run(tmp_path, c, yes=False) is None
    out = capsys.readouterr().out
    assert "3 answer calls" in out and "Dry run" in out
    assert c.calls == [] and not (tmp_path / "answers.json").exists()


def test_estimate_for_three_setups_of_the_real_test_split():
    # 72 test questions x 3 setups = 216 answer calls; 60 answerable x 3 = 180 judge calls
    assert 5 <= estimate_usd(216, 180) <= 10


def test_budget_stop_writes_partial_results(tmp_path):
    c = FakeClient(tokens=(1_000_000, 0))  # $4 per call
    data = _run(tmp_path, c, max_usd=5.0)
    # q1 answer ($4) + judge ($4) = $8 >= $5 -> stops after the first question
    assert data["stopped"] == "budget"
    assert [r["id"] for r in data["setups"][SETUP]["rows"]] == ["q1"]
    saved = json.loads((tmp_path / "answers.json").read_text())
    assert saved["stopped"] == "budget" and len(saved["setups"][SETUP]["rows"]) == 1


def test_resume_skips_finished_questions(tmp_path):
    c1 = FakeClient(tokens=(1_000_000, 0))
    _run(tmp_path, c1, max_usd=5.0)  # stops after q1
    c2 = FakeClient()
    data = _run(tmp_path, c2)
    ids = [r["id"] for r in data["setups"][SETUP]["rows"]]
    assert ids == ["q1", "q2", "q3"]
    asked = [call["messages"][0]["content"][-1]["text"] for call in c2.calls if "format" not in call["output_config"]]
    assert asked == ["Question q2?", "Question q3?"]  # q1 was not asked again
    assert data["stopped"] is None


def test_error_rows_are_recorded_and_retried(tmp_path):
    data = _run(tmp_path, FakeClient(fail=["q2"]))
    rows = {r["id"]: r for r in data["setups"][SETUP]["rows"]}
    assert rows["q2"]["status"] == "error" and "upstream_error" in rows["q2"]["error"]
    assert data["setups"][SETUP]["metrics"]["errors"] == 1
    c2 = FakeClient()
    data = _run(tmp_path, c2)
    rows = {r["id"]: r for r in data["setups"][SETUP]["rows"]}
    assert rows["q2"]["status"] == "answered" and len(data["setups"][SETUP]["rows"]) == 3
    assert sum("format" not in c["output_config"] for c in c2.calls) == 1  # only q2 reran


def test_report_adds_the_answer_table(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    (res / "retrieval.json").write_text("[]")
    data = _run(tmp_path, FakeClient())
    (res / "answers.json").write_text(json.dumps(data))
    text = evalrun.render_block(res)
    assert "| Setup | Answerable / unanswerable | Grounded |" in text and SETUP in text
    assert "Total cost: $" in answers_table(data)
