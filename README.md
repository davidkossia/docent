# Docent

Docent answers questions over technical documentation and cites the exact passages it used. When the documentation does not contain the answer, it says so.

Docent is a 2026 rebuild of CourseVector, an earlier project of mine whose repository was lost. This code is new. It is not the original.

**Status:** under construction. See [PLAN.md](PLAN.md) for milestones. Sections marked "pending" fill in as each milestone lands.

## What it does

- Ingests about 1,000 pages of openly licensed technical text, splits it into chunks, embeds them, and indexes them.
- Retrieves the top passages for a question (LangChain and Pinecone).
- Asks Claude to answer from those passages only. Passages go in as `document` blocks with citations enabled, so each claim links to its source passage.
- Refuses when the passages do not answer the question.
- Serves the answer through an API (AWS Lambda behind API Gateway) to a React chat UI with citation chips that open the source passage.

## Architecture

```
PDF corpus --> loader (per page) --> splitter --> embedder --> vector store
                                                                  |
browser --> API Gateway --> Lambda: embed query --> top-k chunks -+
                                  |
                                  +--> Claude (documents + citations) --> answer
```

| Part | Where | Stack |
|---|---|---|
| Ingest, retrieval, answers, eval | `backend/` | Python, LangChain, Pinecone, Anthropic SDK |
| Infrastructure | `infra/` | AWS CDK (TypeScript) |
| Chat UI | `web/` | React, TypeScript, Vite |

## Eval results

<!-- eval:start -->
Evidence recall@k and MRR@5 by split. Won / lost counts test questions answered in the top 5 that the `fixed-1000` baseline with the same embedder misses (won) or answers (lost).

| Chunking | Embedding | Chunks | Dev R@5 | Dev MRR@5 | Test R@1 | Test R@3 | Test R@5 (95% CI) | Test MRR@5 | Test section R@5 | Won / lost vs baseline |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| fixed-1000 | local-bge-small | 3,653 | 66.7% | 0.532 | 35.0% | 43.3% | 51.7% (38.3% to 65.0%) | 0.407 | 73.3% | - |
| fixed-800 | local-bge-small | 4,125 | 56.7% | 0.466 | 36.7% | 51.7% | 53.3% (40.0% to 65.0%) | 0.435 | 73.3% | 7 / 6 |
| clean-800 | local-bge-small | 3,917 | 58.3% | 0.470 | 31.7% | 43.3% | 50.0% (36.7% to 61.7%) | 0.383 | 78.3% | 8 / 9 |
| section-800-nohdr | local-bge-small | 3,988 | 61.7% | 0.489 | 38.3% | 43.3% | 46.7% (35.0% to 58.4%) | 0.411 | 71.7% | 5 / 8 |
| section-800 (selected) | local-bge-small | 3,988 | 68.3% | 0.491 | 38.3% | 45.0% | 53.3% (40.0% to 65.0%) | 0.435 | 78.3% | 8 / 7 |
| section-500 | local-bge-small | 6,577 | 61.7% | 0.515 | 38.3% | 48.3% | 55.0% (41.7% to 68.3%) | 0.437 | 68.3% | 7 / 5 |
| section-1200 | local-bge-small | 2,780 | 65.0% | 0.501 | 35.0% | 51.7% | 58.3% (45.0% to 70.0%) | 0.439 | 73.3% | 7 / 3 |
| fixed-1000 | local-bge-base | 3,653 | 66.7% | 0.551 | 33.3% | 53.3% | 61.7% (48.3% to 73.3%) | 0.442 | 78.3% | - |
| fixed-800 | local-bge-base | 4,125 | 66.7% | 0.535 | 36.7% | 58.3% | 63.3% (51.7% to 75.0%) | 0.470 | 76.7% | 7 / 6 |
| clean-800 | local-bge-base | 3,917 | 60.0% | 0.468 | 33.3% | 45.0% | 53.3% (40.0% to 66.7%) | 0.403 | 78.3% | 4 / 9 |
| section-800-nohdr | local-bge-base | 3,988 | 61.7% | 0.482 | 26.7% | 50.0% | 53.3% (40.0% to 66.7%) | 0.371 | 78.3% | 6 / 11 |
| section-800 | local-bge-base | 3,988 | 66.7% | 0.514 | 30.0% | 51.7% | 56.7% (43.3% to 68.3%) | 0.402 | 80.0% | 7 / 10 |
| section-500 | local-bge-base | 6,577 | 65.0% | 0.522 | 41.7% | 55.0% | 55.0% (43.3% to 68.3%) | 0.478 | 75.0% | 7 / 11 |
| section-1200 | local-bge-base | 2,780 | 66.7% | 0.496 | 28.3% | 55.0% | 65.0% (53.3% to 76.7%) | 0.424 | 78.3% | 6 / 4 |
<!-- eval:end -->

## Repository layout

```
backend/   Python package, tests, eval data (ingest and chunking exist; eval and API pending)
web/       Vite + React + TypeScript chat UI (placeholder page)
infra/     AWS CDK app (empty stack; Lambda and API Gateway pending)
```

## Run it

Available now:

```bash
cd backend
uv venv --python 3.12 && uv pip install -e ".[local,dev]"
.venv/bin/python -m docent download   # fetches the two PDFs into backend/data/raw
.venv/bin/python -m pytest
```

```bash
cd web && npm install && npm run build
cd infra && npm install && npx cdk synth -c allowedOrigin=http://localhost:5173
```

The eval, API, and UI commands land with their milestones.

## Keys

Copy `.env.example` to `.env` and fill it in. `.env` is git-ignored. Nothing in this repo needs a key to read, and ingest plus the retrieval eval run with no key.

## Corpus and attribution

The corpus is two OpenStax textbooks, downloaded by a script and not stored in this repo:

- *Introduction to Python Programming*, OpenStax, Rice University.
- *Introduction to Computer Science*, OpenStax, Rice University.

Both are licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Docent uses them for non-commercial demonstration only. Download pages: <https://openstax.org/subjects/computer-science>.

## License

Code: [MIT](LICENSE). The textbooks keep their own license.
