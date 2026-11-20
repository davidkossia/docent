"""Eval harness v2: question validation, evidence-hit retrieval metrics, answer eval, report."""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .chunking import STRATEGIES, build_chunks
from .config import CLAUDE_MODEL, EVAL_DIR, FALLBACK_BETA, ROOT
from .embeddings import get_embedder
from .index import LocalIndex
from .ingest import Section, clean_text, load_sections
from .pricing import (
    ANSWER_MAX_TOKENS,
    PRICES,
    UNKNOWN_MODEL_PRICES,
    WORST_CASE_CALL_USD,
)

QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
README_PATH = ROOT.parent / "README.md"
BASELINE = "fixed-1000"
EVAL_START, EVAL_END = "<!-- eval:start -->", "<!-- eval:end -->"

TYPES = ["lookup", "paraphrase", "code", "unanswerable-offtopic", "unanswerable-nearmiss"]
UNANSWERABLE = {"unanswerable-offtopic", "unanswerable-nearmiss"}
SPLITS = ["dev", "test"]
# answerable and unanswerable counts per split (each type splits evenly between dev and test)
EXPECTED_PER_SPLIT = {
    "lookup": 20, "paraphrase": 25, "code": 15, "unanswerable-offtopic": 6, "unanswerable-nearmiss": 6,
}
MAX_PER_SECTION = 2
HIT_FRACTION = 0.70
PARAPHRASE_MAX_JACCARD = 0.25
FIELDS = ["id", "split", "type", "question", "answer", "gold_section", "evidence", "must_not_appear"]

STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "in", "on", "at", "to", "for", "from", "by", "with", "without", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "there", "here", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "do", "does", "did", "done", "can", "could", "should", "would", "will", "may", "might", "must", "shall", "i", "you", "he", "she", "we", "they", "me", "my", "your", "our", "their", "them", "his", "her", "not", "no", "yes", "so", "than", "too", "very", "also", "just", "into", "over", "under", "about", "between", "through", "each", "any", "all", "some", "such", "only", "own", "same", "other", "more", "most", "less", "few", "many", "much", "one", "two", "use", "used", "using", "get", "gets"]
)


# ---------- text helpers ----------

def normalize(text: str) -> str:
    """Lowercase, join words hyphenated across a line break, collapse whitespace."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def lcs_len(a: str, b: str) -> int:
    """Length of the longest common substring of a and b."""
    if not a or not b:
        return 0
    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def is_evidence_hit(chunk_text: str, evidence: str) -> bool:
    """A chunk is a hit when it shares a substring of at least 70% of the evidence length."""
    ev = normalize(evidence)
    if not ev:
        return False
    return lcs_len(normalize(chunk_text), ev) * 100 >= round(HIT_FRACTION * 100) * len(ev)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    x, y = content_words(a), content_words(b)
    return len(x & y) / len(x | y) if x | y else 0.0


# ---------- questions ----------

@dataclass
class Question:
    id: str
    split: str
    type: str
    question: str
    answer: str | None
    gold_section: str | None  # None = unanswerable
    evidence: str | None
    must_not_appear: list[str] | None = None


def load_questions(path=QUESTIONS_PATH) -> list[Question]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out.append(Question(**{k: row.get(k) for k in FIELDS}))
    return out


def _paragraphs(sec: Section) -> list[str]:
    """Normalized paragraphs of the section's cleaned text, one list per page, flattened."""
    out = []
    for _, text in sec.pages:
        out.extend(normalize(p) for p in re.split(r"\n\s*\n", clean_text(text)) if p.strip())
    return out


def validate_questions(path=QUESTIONS_PATH, sections: list[Section] | None = None,
                       check_counts: bool = True) -> list[str]:
    """Check every rule in the eval spec. Returns a list of error strings (empty = valid)."""
    errors: list[str] = []
    rows: list[dict] = []
    try:
        with open(path) as f:
            for n, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as e:
                    errors.append(f"line {n}: invalid JSON ({e})")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"line {n}: not a JSON object")
                    continue
                row["_line"] = n
                rows.append(row)
    except OSError as e:
        return [f"cannot read {path}: {e}"]
    if sections is None:
        sections = load_sections()
    by_id = {s.id: s for s in sections}
    corpus = " ".join(normalize(clean_text(t)) for s in sections for _, t in s.pages)
    paras: dict[str, list[str]] = {}

    seen_ids: set[str] = set()
    per_section: Counter = Counter()
    for row in rows:
        qid = row.get("id")
        tag = f"{qid or 'line ' + str(row['_line'])}"
        err = lambda msg, tag=tag: errors.append(f"{tag}: {msg}")
        missing = [k for k in FIELDS if k not in row]
        extra = [k for k in row if k not in FIELDS and k != "_line"]
        if missing:
            err(f"missing fields {missing}")
        if extra:
            err(f"unknown fields {extra}")
        if missing:
            continue
        if not (isinstance(qid, str) and re.fullmatch(r"q\d{3,}", qid)):
            err("id must look like q001")
        elif qid in seen_ids:
            err("duplicate id")
        seen_ids.add(str(qid))
        if row["split"] not in SPLITS:
            err(f"split must be one of {SPLITS}")
        if row["type"] not in TYPES:
            err(f"type must be one of {TYPES}")
            continue
        q = row["question"]
        if not isinstance(q, str) or not 8 <= len(q.split()) <= 30:
            err("question must be a string of 8 to 30 words")
        if row["type"] in UNANSWERABLE:
            for k in ("answer", "gold_section", "evidence"):
                if row[k] is not None:
                    err(f"{k} must be null for an unanswerable question")
            phrases = row["must_not_appear"]
            if not (isinstance(phrases, list) and 1 <= len(phrases) <= 3
                    and all(isinstance(p, str) and p.strip() for p in phrases)):
                err("must_not_appear must be a list of 1 to 3 phrases")
            else:
                for p in phrases:
                    if normalize(p) in corpus:
                        err(f"must_not_appear phrase {p!r} occurs in the corpus")
            continue
        # answerable
        if row["must_not_appear"] is not None:
            err("must_not_appear must be null for an answerable question")
        if not (isinstance(row["answer"], str) and row["answer"].strip()):
            err("answer must be a non-empty string")
        gold, ev = row["gold_section"], row["evidence"]
        if not isinstance(gold, str) or gold not in by_id:
            err(f"gold_section {gold!r} is not a section in the corpus")
            continue
        per_section[gold] += 1
        if row["type"] == "code" and not str(gold).startswith("py-"):
            err("code questions must use a Python-book section (py-...)")
        if not isinstance(ev, str) or not 40 <= len(ev) <= 300:
            err("evidence must be a string of 40 to 300 characters")
            continue
        if gold not in paras:
            paras[gold] = _paragraphs(by_id[gold])
        if not any(normalize(ev) in p for p in paras[gold]):
            err("evidence is not verbatim text from one paragraph on one page of the gold section")
        if row["type"] == "paraphrase":
            j = jaccard(q, ev)
            if j > PARAPHRASE_MAX_JACCARD:
                err(f"paraphrase overlap with evidence is {j:.2f}, above {PARAPHRASE_MAX_JACCARD}")
    for sid, n in per_section.items():
        if n > MAX_PER_SECTION:
            errors.append(f"section {sid} has {n} questions; the limit is {MAX_PER_SECTION}")
    if check_counts:
        have = Counter((r.get("split"), r.get("type")) for r in rows)
        for split in SPLITS:
            for t, n in EXPECTED_PER_SPLIT.items():
                if have[(split, t)] != n:
                    errors.append(f"count: {split}/{t} is {have[(split, t)]}, expected {n}")
        if len(rows) != 144:
            errors.append(f"count: {len(rows)} questions, expected 144")
    return errors


# ---------- retrieval ----------

def section_rank(hits, gold: str) -> int | None:
    """1-based rank of the first hit whose chunk overlaps the gold section (the v1 rule)."""
    for rank, h in enumerate(hits, start=1):
        if gold in h.doc.metadata["section_ids"]:
            return rank
    return None


hit_rank = section_rank  # v1 name, kept for the answer half


def evidence_rank(hits, evidence: str) -> int | None:
    """1-based rank of the first chunk that is an evidence hit."""
    for rank, h in enumerate(hits, start=1):
        if is_evidence_hit(h.doc.page_content, evidence):
            return rank
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mrr_at(ranks: list[int | None], k: int = 5) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def split_metrics(rows: list[dict]) -> dict:
    ev = [r["rank"] for r in rows]
    sec = [r["section_rank"] for r in rows]
    return {
        "n": len(rows),
        "recall@1": round(recall_at(ev, 1), 4),
        "recall@3": round(recall_at(ev, 3), 4),
        "recall@5": round(recall_at(ev, 5), 4),
        "mrr@5": round(mrr_at(ev, 5), 4),
        "section_recall@5": round(recall_at(sec, 5), 4),
    }


def bootstrap_ci(ranks: list[int | None], k: int = 5, resamples: int = 1000, seed: int = 0) -> list[float]:
    """95% bootstrap interval of recall@k over questions."""
    if not ranks:
        return [0.0, 0.0]
    hit = np.array([r is not None and r <= k for r in ranks], dtype=float)
    rng = np.random.default_rng(seed)
    means = hit[rng.integers(0, len(hit), size=(resamples, len(hit)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def won_lost(ranks: list[int | None], base: list[int | None], k: int = 5) -> dict:
    """Questions this setup answers in the top k where the baseline does not (won), and the reverse (lost)."""
    a = [r is not None and r <= k for r in ranks]
    b = [r is not None and r <= k for r in base]
    return {"won": sum(x and not y for x, y in zip(a, b)), "lost": sum(y and not x for x, y in zip(a, b))}


def eval_retrieval(strategy_name: str, embedder_name: str, questions: list[Question],
                   sections: list[Section], k: int = 5, log=print) -> dict:
    answerable = [q for q in questions if q.gold_section and q.evidence]
    docs = build_chunks(STRATEGIES[strategy_name], sections)
    embedder = get_embedder(embedder_name)
    t0 = time.time()
    index = LocalIndex.build_cached(f"{strategy_name}__{embedder_name}", docs, embedder, log=log)
    build_s = time.time() - t0
    per_q = []
    for q in answerable:
        hits = index.search(q.question, k)
        per_q.append({
            "id": q.id, "split": q.split, "type": q.type,
            "rank": evidence_rank(hits, q.evidence),
            "section_rank": section_rank(hits, q.gold_section),
        })
    metrics = {sp: split_metrics([r for r in per_q if r["split"] == sp]) for sp in SPLITS}
    test_ranks = [r["rank"] for r in per_q if r["split"] == "test"]
    return {
        "strategy": strategy_name,
        "embedder": embedder_name,
        "chunks": len(docs),
        "avg_chunk_chars": round(statistics.mean(len(d.page_content) for d in docs)),
        "index_build_s": round(build_s, 1),
        "metrics": metrics,
        "test_recall@5_ci95": bootstrap_ci(test_ranks),
        "vs_baseline": None,
        "questions": per_q,
    }


def add_baseline_comparison(rows: list[dict]) -> None:
    """Fill `vs_baseline` (test split, recall@5) for every setup whose embedder also ran fixed-1000."""
    base = {r["embedder"]: r for r in rows if r["strategy"] == BASELINE}
    for r in rows:
        b = base.get(r["embedder"])
        if b is None or r is b:
            r["vs_baseline"] = None
            continue
        pick = lambda row: {q["id"]: q["rank"] for q in row["questions"] if q["split"] == "test"}
        mine, theirs = pick(r), pick(b)
        ids = sorted(mine)
        r["vs_baseline"] = won_lost([mine[i] for i in ids], [theirs[i] for i in ids])


def select_setup(rows: list[dict]) -> dict | None:
    """Best dev recall@5; ties by dev MRR@5, then fewer chunks."""
    if not rows:
        return None
    return min(rows, key=lambda r: (-r["metrics"]["dev"]["recall@5"], -r["metrics"]["dev"]["mrr@5"], r["chunks"]))


def run_retrieval(strategies: list[str], embedders: list[str], questions_path=QUESTIONS_PATH,
                  out_path=None) -> list[dict]:
    questions = load_questions(questions_path)
    sections = load_sections()
    rows = []
    for e in embedders:
        for s in strategies:
            print(f"== {s} x {e}", flush=True)
            rows.append(eval_retrieval(s, e, questions, sections))
            m = rows[-1]["metrics"]
            print(f"   dev R@5 {m['dev']['recall@5']:.3f}  test R@5 {m['test']['recall@5']:.3f}", flush=True)
    add_baseline_comparison(rows)
    out = Path(out_path) if out_path else RESULTS_DIR / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out}")
    return rows


# ---------- report ----------

def retrieval_table(rows: list[dict]) -> str:
    pick = select_setup(rows)
    lines = [
        ("| Chunking | Embedding | Chunks | Dev R@5 | Dev MRR@5 | Test R@1 | Test R@3 | Test R@5 (95% CI) "
         "| Test MRR@5 | Test section R@5 | Won / lost vs baseline |"),
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for r in rows:
        d, t = r["metrics"]["dev"], r["metrics"]["test"]
        lo, hi = r["test_recall@5_ci95"]
        wl = r.get("vs_baseline")
        name = r["strategy"] + (" (selected)" if r is pick else "")
        wl_text = "-" if wl is None else str(wl["won"]) + " / " + str(wl["lost"])
        lines.append(
            f"| {name} | {r['embedder']} | {r['chunks']:,} | {d['recall@5']:.1%} | {d['mrr@5']:.3f} "
            f"| {t['recall@1']:.1%} | {t['recall@3']:.1%} | {t['recall@5']:.1%} ({lo:.1%} to {hi:.1%}) "
            f"| {t['mrr@5']:.3f} | {t['section_recall@5']:.1%} | {wl_text} |"
        )
    return "\n".join(lines)


def render_block(results_dir=RESULTS_DIR) -> str:
    path = Path(results_dir) / "retrieval.json"
    if not path.exists():
        return "No results yet. Run `python -m docent eval-retrieval`, then `python -m docent report`."
    rows = json.loads(path.read_text())
    block = (
        "Evidence recall@k and MRR@5 by split. Won / lost counts test questions answered in the top 5 "
        f"that the `{BASELINE}` baseline with the same embedder misses (won) or answers (lost).\n\n"
        + retrieval_table(rows)
    )
    ans_path = Path(results_dir) / "answers.json"
    if ans_path.exists():
        block += "\n\n" + answers_table(json.loads(ans_path.read_text()))
    return block


def write_report(readme=README_PATH, results_dir=RESULTS_DIR) -> None:
    """Replace only the text between the eval markers in the README."""
    text = Path(readme).read_text()
    a, b = text.find(EVAL_START), text.find(EVAL_END)
    if a < 0 or b < a:
        raise SystemExit(f"{readme} needs {EVAL_START} and {EVAL_END} markers")
    new = text[: a + len(EVAL_START)] + "\n" + render_block(results_dir) + "\n" + text[b:]
    Path(readme).write_text(new)
    print(f"updated {readme}")


# ---------- answer eval (calls Claude; costs money) ----------

# Estimates for the dry run: 5 passages in, a short cited answer out, thinking tokens included.
EST_ANSWER_CALL_USD = 0.03
EST_JUDGE_CALL_USD = 0.008
ANSWERS_PATH = RESULTS_DIR / "answers.json"

JUDGE_SYSTEM = (
    "You grade an answer against a reference answer. Reply correct=true when the answer states the same "
    "facts as the reference and contradicts nothing in it. Wording, order and extra true detail do not matter. "
    "Reply correct=false when the answer misses the key fact, contradicts the reference, or declines to answer."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
    "additionalProperties": False,
}


def _get(obj, name, default=0):
    value = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return value or default


def attempt_cost(entry, model: str) -> float:
    """USD for one attempt: `entry` holds input_tokens, output_tokens and the two cache token counts."""
    pin, pout, pread, pwrite = PRICES.get(model, UNKNOWN_MODEL_PRICES)
    return (
        _get(entry, "input_tokens") * pin
        + _get(entry, "output_tokens") * pout
        + _get(entry, "cache_read_input_tokens") * pread
        + _get(entry, "cache_creation_input_tokens") * pwrite
    ) / 1e6


def response_cost(resp) -> float:
    """USD for a response. With `usage.iterations` (server-side fallbacks), sum each attempt at its own model's rates."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0.0
    default_model = getattr(resp, "model", None) or CLAUDE_MODEL
    iterations = _get(usage, "iterations", None)
    if iterations:
        return sum(attempt_cost(_get(it, "usage", it), _get(it, "model", None) or default_model) for it in iterations)
    return attempt_cost(usage, default_model)


class _MeteredMessages:
    def __init__(self, meter, real):
        self._meter, self._real = meter, real

    def create(self, **kw):
        resp = self._real.create(**kw)
        self._meter.cost_usd += response_cost(resp)
        return resp


class MeteredClient:
    """Wraps a client and adds up the cost of every `beta.messages.create` call."""

    def __init__(self, client):
        self.cost_usd = 0.0
        beta = type("Beta", (), {})()
        beta.messages = _MeteredMessages(self, client.beta.messages)
        self.beta = beta


def parse_setup(name: str) -> tuple[str, str]:
    """`section-800__local-bge-small` -> (strategy, embedder)."""
    from .embeddings import NAMES

    strategy, sep, embedder = name.partition("__")
    if not sep or strategy not in STRATEGIES or embedder not in NAMES:
        raise SystemExit(f"bad setup {name!r}: use <strategy>__<embedder>, for example section-800__local-bge-small")
    return strategy, embedder


def estimate_usd(n_answer_calls: int, n_judge_calls: int) -> float:
    return n_answer_calls * EST_ANSWER_CALL_USD + n_judge_calls * EST_JUDGE_CALL_USD


def judge_answer(client, question: str, reference: str, answer: str) -> tuple[bool | None, str | None]:
    """Ask the judge whether `answer` matches `reference`.

    Returns (verdict, serving model). The verdict is None when the judge refuses or replies off-schema.
    """
    prompt = f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade: {answer}"
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,  # thinking tokens count against this
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=JUDGE_SYSTEM,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    model = getattr(resp, "model", None)
    if resp.stop_reason == "refusal":
        return None, model
    text = "".join(b.text for b in resp.content if b.type == "text")
    try:
        value = json.loads(text)["correct"]
    except (ValueError, KeyError, TypeError):
        return None, model
    return (value if isinstance(value, bool) else None), model


def evaluate_question(q: Question, body: dict) -> dict:
    """Score one `service.ask` response body against its question (no judge yet)."""
    answerable = q.type not in UNANSWERABLE
    status = body["status"]
    grounded = False
    if answerable and status == "answered":
        passages = body["passages"]
        for seg in body["segments"]:
            for c in seg["citations"]:
                n = c["passage"]
                if 1 <= n <= len(passages) and is_evidence_hit(passages[n - 1]["text"], q.evidence):
                    grounded = True
    text = "".join(seg["text"] for seg in body["segments"]).strip()
    return {
        "id": q.id, "type": q.type, "answerable": answerable, "status": status, "grounded": grounded,
        "correct": None, "judge_model": None, "judge_error": None, "answer": text, "error": None,
        "cost_usd": 0.0,
    }


def answer_metrics(rows: list[dict]) -> dict:
    """The four answer metrics over the rows run so far. Errored rows count as misses."""
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    def rate(hits: int, n: int):
        return round(hits / n, 4) if n else None

    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "judge_errors": sum(1 for r in rows if r.get("judge_error")),
        "grounded_rate": rate(sum(r["status"] == "answered" and r["grounded"] for r in ans), len(ans)),
        "judged_correct_rate": rate(sum(r["correct"] is True for r in ans), len(ans)),
        "refusal_accuracy": rate(sum(r["status"] == "no_answer" for r in unans), len(unans)),
        "wrong_refusal_rate": rate(sum(r["status"] == "no_answer" for r in ans), len(ans)),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }


def _load_answers(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
        data.setdefault("dropped_cost_usd", 0.0)
        return data
    return {"model": CLAUDE_MODEL, "setups": {}, "cost_usd": 0.0, "dropped_cost_usd": 0.0, "stopped": None}


def _total_cost(data: dict) -> float:
    """Every dollar this file has spent: kept rows plus the cost of error rows dropped for a rerun."""
    return round(sum(answer_metrics(e["rows"])["cost_usd"] for e in data["setups"].values())
                 + data["dropped_cost_usd"], 4)


def _save_answers(path: Path, data: dict) -> None:
    for entry in data["setups"].values():
        entry["metrics"] = answer_metrics(entry["rows"])
    data["cost_usd"] = _total_cost(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def _local_retriever(setup: str):
    from .embeddings import get_embedder
    from .service import LocalRetriever

    strategy, embedder_name = parse_setup(setup)
    docs = build_chunks(STRATEGIES[strategy], load_sections())
    index = LocalIndex.build_cached(setup, docs, get_embedder(embedder_name), log=print)
    return LocalRetriever(index)


def _needs_judge(row: dict) -> bool:
    return row["answerable"] and row["status"] == "answered" and row["correct"] is None and bool(row.get("judge_error"))


def run_answers(setups: list[str], max_usd: float, yes: bool, questions_path=QUESTIONS_PATH, out_path=None,
                client=None, retriever_factory=None, k: int = 5) -> dict | None:
    """Answer every test question for each setup, then score. Without `yes`, print the estimate and stop.

    Writes the results file after each question. A rerun skips questions that already have a row, runs only
    the judge for rows whose judge call failed, and runs rows with status `error` again. `max_usd` caps the
    spend recorded in the results file, so earlier runs count against it. The check runs before every call,
    so the total can pass `max_usd` by at most one call (see `WORST_CASE_CALL_USD`).
    """
    import math

    from .service import ServiceError, ask

    if not math.isfinite(max_usd) or max_usd < 0:
        raise SystemExit(f"--max-usd must be a finite number of at least 0, got {max_usd}")
    setups = list(dict.fromkeys(setups))
    for s in setups:
        parse_setup(s)
    questions = [q for q in load_questions(questions_path) if q.split == "test"]
    out = Path(out_path) if out_path else ANSWERS_PATH
    data = _load_answers(out)

    def rows_of(setup):
        return data["setups"].get(setup, {}).get("rows", [])

    todo = {s: [q for q in questions if q.id not in {r["id"] for r in rows_of(s) if r["status"] != "error"}]
            for s in setups}
    n_answer = sum(len(v) for v in todo.values())
    n_judge = sum(1 for v in todo.values() for q in v if q.type not in UNANSWERABLE)
    n_judge += sum(1 for s in setups for r in rows_of(s) if _needs_judge(r))
    prior = _total_cost(data)
    est = estimate_usd(n_answer, n_judge)
    print(f"{len(questions) * len(setups)} answer calls ({len(questions)} test questions x {len(setups)} setups)")
    if n_answer < len(questions) * len(setups):
        print(f"{n_answer} still to run; the rest are in {out}")
    print(f"{n_judge} judge calls at most; estimated cost ${est:.2f} at {CLAUDE_MODEL}")
    print(f"Budget ${max_usd:.2f} covers the ${prior:.2f} already spent in {out}. The check runs before each call, "
          f"so the total can pass it by up to ${WORST_CASE_CALL_USD:.2f} per answer call (one attempt on {CLAUDE_MODEL} "
          f"plus one fallback attempt, each at max_tokens {ANSWER_MAX_TOKENS}).")
    if prior + est > max_usd:
        print("The estimate is over the remaining budget; the run stops cleanly when it reaches the budget.")
    if not yes:
        print("Dry run. Pass --yes to spend it.")
        return None

    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    meter = MeteredClient(client)

    def spent() -> float:
        return prior + meter.cost_usd

    def over_budget() -> bool:
        if spent() >= max_usd:
            data["stopped"] = "budget"
            print(f"Stopped: total spend ${spent():.2f} reached the ${max_usd:.2f} budget. Rerun with a higher "
                  "--max-usd to resume.")
            return True
        return False

    data["stopped"] = None
    _save_answers(out, data)

    def judge(row, q_text, reference):
        verdict, model = judge_answer(meter, q_text, reference, row["answer"])
        row["correct"], row["judge_model"] = verdict, model
        row["judge_error"] = None if verdict is not None else "no verdict (refusal or unparseable reply)"

    stop = False
    for setup in setups:
        if stop:
            break
        entry = data["setups"].setdefault(setup, {"rows": []})
        # Error rows rerun; their cost stays in the file total. Only setups in this run are touched.
        dropped = [r for r in entry["rows"] if r["status"] == "error"]
        entry["rows"] = [r for r in entry["rows"] if r["status"] != "error"]
        data["dropped_cost_usd"] = round(data["dropped_cost_usd"] + sum(r["cost_usd"] for r in dropped), 6)
        ref = {q.id: q for q in questions}
        pending = [r for r in entry["rows"] if _needs_judge(r)]
        if not todo[setup] and not pending:
            continue
        print(f"== {setup}", flush=True)
        for row in pending:
            if over_budget():
                stop = True
                break
            before = meter.cost_usd
            try:
                judge(row, ref[row["id"]].question, ref[row["id"]].answer)
            except Exception as exc:  # noqa: BLE001 - the row records the failure
                row["judge_error"] = f"{type(exc).__name__}: {exc}"
            row["cost_usd"] = round(row["cost_usd"] + meter.cost_usd - before, 6)
            _save_answers(out, data)
            print(f"   {row['id']} judge retry correct={row['correct']} spent ${spent():.3f}", flush=True)
        if stop:
            break
        retriever = None
        for q in todo[setup]:
            if over_budget():
                stop = True
                break
            retriever = retriever or (retriever_factory or _local_retriever)(setup)
            before = meter.cost_usd
            try:
                body = ask(q.question, retriever, meter, k=k)
                row = evaluate_question(q, body)
            except ServiceError as exc:
                row = evaluate_question(q, {"status": "error", "segments": [], "passages": []})
                row["error"] = f"{exc.code}: {exc.message}"
            except Exception as exc:  # noqa: BLE001 - keep the run going; the row records the failure
                row = evaluate_question(q, {"status": "error", "segments": [], "passages": []})
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["cost_usd"] = round(meter.cost_usd - before, 6)
            answered = row["answerable"] and row["status"] == "answered"
            if answered:
                row["judge_error"] = "not run: interrupted"  # a rerun judges the row if we stop before the verdict
            entry["rows"].append(row)
            _save_answers(out, data)  # the paid answer is on disk before the judge call starts
            if answered:
                if over_budget():
                    row["judge_error"] = "not run: budget reached"
                    stop = True
                else:
                    mid = meter.cost_usd
                    try:
                        judge(row, q.question, q.answer)
                    except Exception as exc:  # noqa: BLE001
                        row["judge_error"] = f"{type(exc).__name__}: {exc}"
                    row["cost_usd"] = round(row["cost_usd"] + meter.cost_usd - mid, 6)
                _save_answers(out, data)
            print(f"   {q.id} {row['status']} grounded={row['grounded']} correct={row['correct']} "
                  f"spent ${spent():.3f}", flush=True)
            if stop:
                break
    _save_answers(out, data)
    print(f"wrote {out}; total cost ${data['cost_usd']:.2f}")
    return data


def answers_table(data: dict) -> str:
    def pct(v):
        return "-" if v is None else f"{v:.1%}"

    lines = [
        f"Answer eval on the test split, judged by `{data.get('model', CLAUDE_MODEL)}`. "
        "Grounded: answered with a citation to a chunk that holds the evidence. Wrong refusal: answerable "
        "question answered with no_answer."
        + (" The run stopped at its budget; rows are partial." if data.get("stopped") else ""),
        "",
        "| Setup | Answerable / unanswerable | Grounded | Judged correct | Refusal accuracy | Wrong refusal | Errors | Judge errors | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, e in data["setups"].items():
        m = e["metrics"]
        lines.append(
            f"| {name} | {m['n_answerable']} / {m['n_unanswerable']} | {pct(m['grounded_rate'])} "
            f"| {pct(m['judged_correct_rate'])} | {pct(m['refusal_accuracy'])} | {pct(m['wrong_refusal_rate'])} "
            f"| {m['errors']} | {m['judge_errors']} | ${m['cost_usd']:.2f} |"
        )
    lines.append(f"\nTotal cost: ${data['cost_usd']:.2f}")
    return "\n".join(lines)
