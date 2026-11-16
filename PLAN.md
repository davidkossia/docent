# Docent plan

Docent answers questions over technical documentation and cites the passages it used.
It is a 2026 rebuild of CourseVector, an earlier project of mine whose repository was lost.

## Goal

A public repo that runs and shows the work behind three resume claims:

1. A RAG (retrieval-augmented generation) pipeline with LangChain and Pinecone over 500+ pages of unstructured documentation.
2. A tuned chunking and embedding strategy that raises retrieval quality and cuts hallucinations.
3. A serverless backend on AWS Lambda behind API Gateway, with a TypeScript and React front end.

## Decisions

| Area | Choice | Reason |
|---|---|---|
| Corpus | Two OpenStax books: *Introduction to Python Programming* and *Introduction to Computer Science* (about 1,000 PDF pages) | Openly licensed (CC BY-NC-SA 4.0), real page numbers, 500+ pages |
| Loading and splitting | LangChain `Document` and text splitters | Matches the original stack |
| Vector store | Pinecone serverless through `langchain-pinecone`; LangChain `InMemoryVectorStore` for local runs and eval | Eval and the demo run with no Pinecone key |
| Embeddings | One `Embedder` interface with three providers: `openai` (`text-embedding-3-small`, the original's choice and the production default), `voyage`, and `local` (fastembed, ONNX, no key) | The eval compares providers; the local provider lets anyone run it |
| Answers | Claude `claude-opus-5-5` through the Anthropic Python SDK. Retrieved chunks go in as `document` blocks with `citations` enabled. | Every answer cites its passages |
| Refusals | The system prompt tells the model to refuse when the passages lack the answer. A grader counts refusals. | Cuts hallucination |
| Safety fallback | `fallbacks: "default"` with beta `server-side-fallback-2026-07-01` | Opus 5.5 can decline on a safety classifier; the API reruns the request on a fallback model |
| Backend | Python 3.12 on AWS Lambda behind API Gateway (HTTP API), defined in AWS CDK (TypeScript) | Serverless, as in the original |
| Front end | React + TypeScript (Vite). Citation chips open the source passage. | |

## Architecture

```
PDF corpus --> loader (per page) --> splitter (strategy) --> embedder --> vector store
                                                                              |
browser (React) --> API Gateway --> Lambda: embed query --> top-k chunks -----+
                                          |
                                          +--> Claude (document blocks + citations) --> answer + citations
```

Offline ingest runs on a laptop and writes the index (Pinecone or a local file). The Lambda only does the query path.

## Eval design

- **Question set:** 40+ labeled questions in `backend/eval/questions.jsonl`. Each answerable question names a gold source section (book, section number, page range). About 15% are unanswerable (the corpus has no answer); they test refusals.
- **Setups compared** (at least three), varying chunking and embedding:
  - `fixed-1000`: recursive splitter, 1,000 characters, 200 overlap
  - `fixed-400`: recursive splitter, 400 characters, 60 overlap
  - `section-800`: split on detected section headings first, then 800 characters, 100 overlap, with the section title prepended to each chunk
  - each with a local embedding model, plus the `section-800` winner with OpenAI `text-embedding-3-small` once a key exists
- **Metrics:**
  - recall@5: the share of answerable questions where at least one of the top 5 chunks falls inside the gold section
  - grounded-answer rate: the share of answerable questions where Claude's answer carries at least one citation to a chunk inside the gold section
  - refusal accuracy: the share of unanswerable questions where Claude refuses instead of answering
- Recall@5 runs with no keys. The two Claude metrics need `ANTHROPIC_API_KEY` and cost money, so they wait for David's yes.

## Milestones

1. Plan, README, license. (this commit)
2. Ingest and chunking: download, per-page load, section detection, three splitters, tests.
3. Eval harness: question set, retrieval recall@5 across setups, results table in the README.
4. API: retrieval + Claude answer with citations and refusals; Lambda handler; local dev server; CDK stack that synthesizes.
5. UI: chat with citation chips and a source-passage panel.
6. Showable: README with results, local demo, portfolio card.
7. Deploy (optional, needs David's yes).

## Costs

| Item | Estimate | Note |
|---|---|---|
| Local embedding | $0 | fastembed runs on the laptop |
| OpenAI embeddings | under $0.05 per full corpus | about 1M tokens at $0.02 per 1M |
| Pinecone serverless | $0 on the Starter plan | one index, under the free storage limit |
| Claude per question | about $0.02 | Opus 5.5 at $4 / $20 per 1M tokens, 5 passages, low effort |
| Claude eval run | about $1 to $2 per setup | about 50 questions |
| Lambda + API Gateway | $0 at demo volume | inside the free tier |

## Needs David's keys or a yes

- `ANTHROPIC_API_KEY` in `.env`, plus a yes to spend about $5 on the Claude half of the eval.
- `OPENAI_API_KEY` for the OpenAI embedding setup (optional; the local setups run without it).
- `PINECONE_API_KEY` and a yes before I create the index.
- AWS credentials and a yes before `cdk deploy`.
