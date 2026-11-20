"""Eval harness v2: question validation, evidence-hit retrieval metrics, report.

The Claude half (answer eval) is at the bottom of this file and belongs to a later work package.
"""
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
from .config import EVAL_DIR, ROOT
from .embeddings import get_embedder
from .index import LocalIndex
from .ingest import Section, clean_text, load_sections

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
    return (
        "Evidence recall@k and MRR@5 by split. Won / lost counts test questions answered in the top 5 "
        f"that the `{BASELINE}` baseline with the same embedder misses (won) or answers (lost).\n\n"
        + retrieval_table(rows)
    )


def write_report(readme=README_PATH, results_dir=RESULTS_DIR) -> None:
    """Replace only the text between the eval markers in the README."""
    text = Path(readme).read_text()
    a, b = text.find(EVAL_START), text.find(EVAL_END)
    if a < 0 or b < a:
        raise SystemExit(f"{readme} needs {EVAL_START} and {EVAL_END} markers")
    new = text[: a + len(EVAL_START)] + "\n" + render_block(results_dir) + "\n" + text[b:]
    Path(readme).write_text(new)
    print(f"updated {readme}")


# --- Claude half: grounded-answer rate and refusals. Costs money; needs ANTHROPIC_API_KEY. ---

COST_PER_QUESTION_USD = 0.03  # Opus 5.5 at $4/$20 per 1M tokens, 5 passages, low effort (estimate)


def eval_answers(strategy_name: str, embedder_name: str, k: int = 5, log=print) -> dict:
    import anthropic

    from .answer import answer_question

    questions = load_questions()
    docs = build_chunks(STRATEGIES[strategy_name], load_sections())
    index = LocalIndex.build_cached(f"{strategy_name}__{embedder_name}", docs, get_embedder(embedder_name), log=log)
    client = anthropic.Anthropic()
    rows = []
    for q in questions:
        hits = index.search(q.question, k)
        a = answer_question(client, q.question, hits)
        cited = {c.section_id for c in a.citations}
        in_gold = bool(q.gold_section) and any(q.gold_section in hits[c.rank - 1].doc.metadata["section_ids"] for c in a.citations)
        rows.append({
            "id": q.id, "answerable": bool(q.gold_section), "no_answer": a.no_answer,
            "safety_refusal": a.safety_refusal, "n_citations": len(a.citations),
            "cites_gold": in_gold, "cited_sections": sorted(cited), "answer": a.text, "usage": a.usage,
        })
        log(f"{q.id} answerable={bool(q.gold_section)} no_answer={a.no_answer} cites_gold={in_gold}")
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "strategy": strategy_name, "embedder": embedder_name,
        "grounded_rate": round(sum(r["cites_gold"] and not r["no_answer"] for r in ans) / len(ans), 3),
        "refusal_accuracy": round(sum(r["no_answer"] for r in unans) / len(unans), 3),
        "wrong_refusal_rate": round(sum(r["no_answer"] for r in ans) / len(ans), 3),
        "rows": rows,
    }


def run_answers(strategies: list[str], embedders: list[str], yes: bool) -> None:
    n = len(load_questions()) * len(strategies) * len(embedders)
    print(f"{n} Claude calls, about ${n * COST_PER_QUESTION_USD:.2f} at claude-opus-5-5")
    if not yes:
        print("Dry run. Pass --yes to spend it.")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for e in embedders:
        for s in strategies:
            out.append(eval_answers(s, e))
    (RESULTS_DIR / "answers.json").write_text(json.dumps(out, indent=2) + "\n")
    for r in out:
        print(r["strategy"], r["embedder"], "grounded", r["grounded_rate"], "refusal", r["refusal_accuracy"])
