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
Pending. Run `python -m docent eval-retrieval`, then `python -m docent report`.
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
cd infra && npm install && npx cdk synth
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
