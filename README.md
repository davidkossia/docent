# Docent

Docent answers questions over technical documentation and cites the exact passages it used. When the documentation does not contain the answer, it says so.

Docent is a 2026 rebuild of CourseVector, an earlier project of mine whose repository was lost. This code is new. It is not the original.

## What it does

- Loads two OpenStax textbooks (1,256 content pages), splits them into chunks, embeds the chunks, and indexes them.
- Retrieves the top passages for a question with LangChain.
- Asks Claude to answer from those passages only. The passages go in as `document` blocks with citations enabled, so each claim links to its source passage.
- Refuses when the passages do not answer the question.
- Serves the answer through `POST /ask` ([contract](docs/api.md)) to a React chat UI. Citation chips open the source passage.

## Status

| Part | State |
|---|---|
| Ingest and chunking (seven strategies) | Done |
| Retrieval eval: 144 questions, 14 setups | Done. Results follow. |
| Local API server and chat UI | Done. Runs locally. |
| Answer eval with Claude | Not run. It needs an Anthropic key and a $10 spending cap. |
| Pinecone index | Not created. The `index` command exists and needs a Pinecone key. |
| AWS deploy (Lambda, API Gateway, daily question cap) | Built and synthesized. Not deployed. |

No live demo exists. The retrieval numbers below are the only measured results.

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
| Ingest, retrieval, answers, eval | `backend/` | Python, LangChain, Anthropic SDK, Pinecone |
| Infrastructure | `infra/` | AWS CDK (TypeScript) |
| Chat UI | `web/` | React, TypeScript, Vite |

Locally, the index is a file of vectors and the embedder is a local ONNX model, so retrieval needs no key. The planned deployment uses Pinecone and a Pinecone-hosted embedder.

## Eval results

The eval measures retrieval only: whether the passage that holds the answer reaches the model. It has 144 questions written by Claude from the book text: 120 answerable (40 `lookup`, 50 `paraphrase`, 30 `code`) and 24 unanswerable (12 off-topic, 12 near-miss). Each answerable question stores a verbatim evidence quote from its gold section. The questions split evenly into dev (72) and test (72); the 60 answerable test questions set the recall figures. A retrieved chunk is a hit when it contains at least 70% of the evidence quote, so a hit means the answer text arrived, not only the right section. The selection rule picks the setup with the best dev recall@5 and reports every setup on test, with no changes to questions or setups after test numbers existed. Limits: 60 test questions give wide intervals (about 13 points either side), and a model wrote the questions, so they can favor the wording of the text. The `paraphrase` type and its word-overlap check reduce that bias but do not remove it. Full design: [PLAN.md](PLAN.md).

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

At 60 test questions, no chunking or embedder change is statistically significant. The selected setup, `section-800` with `local-bge-small`, scores 53.3% test recall@5 against 51.7% for the `fixed-1000` baseline, and the intervals overlap almost entirely. The embedder gap is the larger lead: `local-bge-base` scores higher than `local-bge-small` with six of the seven chunking strategies and ties on the seventh, but those intervals overlap too, so it is a lead to test with more questions and not a result. This eval does not show that tuning raised retrieval quality.

## Repository layout

```
backend/   Python package (docent/), tests, eval questions and results
web/       Vite + React + TypeScript chat UI
infra/     AWS CDK app: Lambda, API Gateway, DynamoDB daily counter
docs/      API contract
```

## Run it

You need Python 3.12 with [uv](https://docs.astral.sh/uv/), and Node 22. Run each block from the repository root.

Set up the backend and download the corpus (two PDFs, about 1,000 pages, saved to `backend/data/raw`):

```bash
cd backend
uv venv --python 3.12 && uv pip install -e ".[local,dev]"
.venv/bin/python -m docent download
```

Run the tests and the question-file check. Neither needs a key:

```bash
cd backend
.venv/bin/python -m pytest
.venv/bin/python -m docent eval-validate
```

Run the retrieval eval and rewrite the results table in this README. The first run embeds every chunk with local models and takes a long time; later runs reuse the cached indexes in `backend/data/indexes`. Narrow the run with `--strategies` and `--embedders` (see `--help`).

```bash
cd backend
.venv/bin/python -m docent eval-retrieval
.venv/bin/python -m docent report
```

Run the API on `http://localhost:8787/ask` and the chat UI on `http://localhost:5173`. The UI reads `VITE_API_URL` and defaults to the local API.

```bash
# Terminal 1. --fake answers from web/src/fixtures and calls no model.
cd backend
.venv/bin/python -m docent serve --fake

# Terminal 2
cd web
npm ci
npm run dev
```

To answer with Claude instead, copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Build the default local index once (`section-800` with `local-bge-small`), then drop `--fake`. Each question costs about $0.02.

```bash
cd backend
.venv/bin/python -m docent eval-retrieval --strategies section-800 --embedders local-bge-small
.venv/bin/python -m docent serve
```

Test and build the UI and the infrastructure code:

```bash
cd web && npm ci && npm run build && npm test
cd infra && npm ci && npm run build && npm test
```

Synthesize the CloudFormation template. The CDK app needs the Lambda bundle first, which `build_lambda.sh` builds from committed code with `uv`. This does not deploy anything.

```bash
infra/scripts/build_lambda.sh
cd infra && npx cdk synth -c allowedOrigin=http://localhost:5173
```

## Keys

`.env` is git-ignored. Nothing here needs a key to read. Tests, `eval-validate`, `eval-retrieval`, `report`, `serve --fake`, and the UI run with no key. `serve` needs `ANTHROPIC_API_KEY`. `index` needs `PINECONE_API_KEY`. `eval-answers` calls Claude, costs money, and needs `--yes`.

## Corpus and attribution

The corpus is two OpenStax textbooks. A script downloads them and this repo does not store them:

- *Introduction to Python Programming*, OpenStax, Rice University.
- *Introduction to Computer Science*, OpenStax, Rice University.

Both are licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Docent uses them for non-commercial demonstration only. Download pages: <https://openstax.org/subjects/computer-science>.

## License

Code: [MIT](LICENSE). The textbooks keep their own license, CC BY-NC-SA 4.0.
