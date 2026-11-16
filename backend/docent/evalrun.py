"""Eval harness: retrieval recall@k across chunking x embedding setups."""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass

from .chunking import STRATEGIES, build_chunks
from .config import EVAL_DIR
from .embeddings import get_embedder
from .index import LocalIndex
from .ingest import load_sections

QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


@dataclass
class Question:
    id: str
    question: str
    gold_section: str | None  # None = unanswerable
    answer: str | None


def load_questions(path=QUESTIONS_PATH) -> list[Question]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                out.append(Question(f"q{i + 1:03d}", row["question"], row["gold_section"], row["answer"]))
    return out


def hit_rank(hits, gold: str) -> int | None:
    """1-based rank of the first hit whose chunk overlaps the gold section."""
    for rank, h in enumerate(hits, start=1):
        if gold in h.doc.metadata["section_ids"]:
            return rank
    return None


def eval_retrieval(strategy_name: str, embedder_name: str, k: int = 5, log=print) -> dict:
    questions = [q for q in load_questions() if q.gold_section]
    strategy = STRATEGIES[strategy_name]
    docs = build_chunks(strategy, load_sections())
    embedder = get_embedder(embedder_name)
    t0 = time.time()
    index = LocalIndex.build_cached(f"{strategy_name}__{embedder_name}", docs, embedder, log=log)
    build_s = time.time() - t0
    ranks, misses = [], []
    for q in questions:
        r = hit_rank(index.search(q.question, k), q.gold_section)
        ranks.append(r)
        if r is None:
            misses.append(q.id)
    n = len(questions)
    return {
        "strategy": strategy_name,
        "embedder": embedder_name,
        "chunks": len(docs),
        "avg_chunk_chars": round(statistics.mean(len(d.page_content) for d in docs)),
        "questions": n,
        f"recall@{k}": round(sum(r is not None for r in ranks) / n, 3),
        "recall@1": round(sum(r == 1 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "missed": misses,
        "index_build_s": round(build_s, 1),
    }


def retrieval_table(rows: list[dict]) -> str:
    lines = [
        "| Chunking | Embedding | Chunks | Avg chars | Recall@1 | Recall@5 | MRR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['strategy']} | {r['embedder']} | {r['chunks']:,} | {r['avg_chunk_chars']} "
            f"| {r['recall@1']:.0%} | {r['recall@5']:.0%} | {r['mrr']:.3f} |"
        )
    return "\n".join(lines)


def run_retrieval(strategies: list[str], embedders: list[str]) -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in embedders:
        for s in strategies:
            print(f"== {s} x {e}")
            rows.append(eval_retrieval(s, e))
            print(retrieval_table(rows[-1:]))
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps(rows, indent=2) + "\n")
    (RESULTS_DIR / "retrieval.md").write_text(retrieval_table(rows) + "\n")
    return rows


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
