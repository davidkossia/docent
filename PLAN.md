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
| Embeddings | One `Embedder` interface: a Pinecone-hosted model (production), `local` (fastembed, ONNX, no key; used for eval and local runs), and optional `openai` (`text-embedding-3-small`, the original's choice) and `voyage` | Production needs only the Anthropic and Pinecone keys; the local provider lets anyone run the eval |
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

## Status (2026-11-20)

| Milestone | State |
|---|---|
| Plan, README, license | Done |
| Repo scaffold: `backend/`, `web/`, `infra/`, CI | Done |
| Ingest and chunking | Done: 1,256 content pages, 203 sections, seven chunking strategies |
| Eval version 2: harness and 144 questions | Done |
| Retrieval grid on local embedders | Done: 14 setups; results in the README |
| Service, local server, chat UI | Done; runs locally with `serve --fake`, or with an Anthropic key |
| Pinecone index command | Done; not run (needs a Pinecone key) |
| Lambda, CDK stack, spend cap | Done; synthesized, not deployed |
| Answer eval with Claude | Code in review; not run (needs a key and a $10 cap) |

### Retrieval result: no change is significant at 60 test questions

The selection rule picks `section-800` with `local-bge-small`, the best dev recall@5 (68.3%). On test it reaches 53.3% recall@5 against 51.7% for the `fixed-1000` baseline with the same embedder: 8 questions won, 7 lost. The 95% intervals overlap almost entirely, so no chunking change separates from noise at 60 test questions.

The embedder gap is larger than any chunking gap, though its intervals also overlap, so it is a lead and not a result. With `fixed-1000` chunking, `local-bge-base` reaches 61.7% test recall@5 against 51.7% for `local-bge-small`, and its best setup (`section-1200`) reaches 65.0%. These numbers come from the test split, so they inform the next step but do not change the selection. The deployed service uses a Pinecone-hosted embedder, measured once a Pinecone key exists.

### Finding: eval version 1 cannot tell setups apart

Version 1 has 48 answerable questions and counts a hit when any top-5 chunk comes from the right section. With the local `bge-small` embedder, all four chunking setups score 48 of 48. Recall@1 ranges from 37 to 42 of 48, which is inside the noise for 48 questions. The hit rule is too loose (sections run 3 to 20 pages) and the questions reuse the source wording. Version 2 fixes both.

## Eval design (version 2)

**Questions.** `backend/eval/questions.jsonl`: 120 answerable and 24 unanswerable, split evenly into dev and test.

| Type | Count | What it tests |
|---|---:|---|
| `lookup` | 40 | A stated fact; may reuse the source's terms |
| `paraphrase` | 50 | The same need in different words (low word overlap with the source, checked by a validator) |
| `code` | 30 | What a piece of Python does, or which construct to use |
| `unanswerable-offtopic` | 12 | A technical question the books do not cover |
| `unanswerable-nearmiss` | 12 | The topic is in the books, the specific answer is not |

Each answerable question stores its gold section and an `evidence` quote copied verbatim from that section.

**Hit rule.** A retrieved chunk is a hit when it contains at least 70% of the evidence quote (longest common substring after normalizing case, whitespace, and line-break hyphens). This measures whether the answer text reached the model, not only whether the right section did.

**Chunking ladder.** One change per step, so the table shows which change helps:

| Name | Size, overlap | Cleaned text | Split inside sections | Title header |
|---|---|---|---|---|
| `fixed-1000` (baseline) | 1000, 200 | no | no | no |
| `fixed-800` | 800, 100 | no | no | no |
| `clean-800` | 800, 100 | yes | no | no |
| `section-800-nohdr` | 800, 100 | yes | yes | no |
| `section-800` | 800, 100 | yes | yes | yes |
| `section-500` | 500, 80 | yes | yes | yes |
| `section-1200` | 1200, 150 | yes | yes | yes |

Embedders: `local-bge-small` and `local-bge-base` (no key), then a hosted model through Pinecone Inference, and OpenAI `text-embedding-3-small` if a key exists.

**Retrieval metrics.** Evidence recall@1, @3, @5 and MRR@5 on dev and on test; a 95% bootstrap interval on test recall@5; questions won and lost against the baseline.

**Selection rule.** Choose the setup on dev. Report every setup on test. No changes to questions, metrics, or setups after test numbers exist. The README reports the result as measured, including if the tuned setup does not beat the baseline.

**Answer metrics (Claude, test split).** For the baseline, the best chunking, and the selected setup:

- grounded-answer rate: the answer cites at least one chunk that contains the evidence
- judged-correct rate: a second Claude call compares the answer with the reference answer
- refusal accuracy on unanswerable questions, and wrong-refusal rate on answerable ones

## Design changes since the first plan

- **Production embedder:** a Pinecone-hosted model through `langchain_pinecone.PineconeEmbeddings`. The Lambda then needs two keys (Anthropic, Pinecone) and no local model.
- **Spend protection:** the public endpoint gets a daily question cap (an atomic DynamoDB counter), API Gateway throttling, a Lambda concurrency limit, and keys in SSM Parameter Store. Nothing deploys without these.
- **API contract first:** `POST /ask` returns answer segments, each with its citations, plus the retrieved passages. The UI builds against fixtures while the service is written. The contract lands in `docs/api.md`.
- **Answer call:** `max_tokens` rises to 16,000 because thinking tokens count against it, and the `max_tokens` stop reason is handled.

## Milestones

1. Eval core version 2: schema, validator, evidence-hit metric, chunking ladder, report command.
2. Question set version 2, passing the validator.
3. Retrieval grid on local embedders; pick the setup on dev; results table in the README.
4. Service and local server that return the API contract; chat UI against fixtures.
5. Answer eval with Claude (needs a key and a spending yes).
6. Pinecone index command; Lambda, CDK stack, and spend cap, built and synthesized.
7. README final, portfolio card.
8. Deploy (optional, needs a yes).

## Costs

| Item | Estimate | Note |
|---|---|---|
| Local embedding | $0 | fastembed runs on the laptop |
| Pinecone-hosted embeddings | $0 expected | about 1M tokens per setup; the free allowance is checked before use |
| OpenAI embeddings (optional) | under $0.05 per full corpus | about 1M tokens at $0.02 per 1M |
| Pinecone serverless | $0 on the Starter plan | one index, under the free storage limit |
| Claude per question | about $0.02 | Opus 5.5 at $4 / $20 per 1M tokens, 5 passages, low effort |
| Claude answer eval | about $7 to $8, hard cap $10 | 216 answer calls and about 180 judge calls on the test split |
| Lambda + API Gateway | $0 at demo volume | inside the free tier |

## Needs David's keys or a yes

- `ANTHROPIC_API_KEY` in `.env`, plus a yes to a $10 cap for the answer eval.
- `OPENAI_API_KEY` for the optional OpenAI eval row.
- `PINECONE_API_KEY` and a yes before I create the index.
- AWS credentials, the two keys stored in SSM Parameter Store, a spend limit on the Anthropic workspace, and a yes before `cdk deploy`.
