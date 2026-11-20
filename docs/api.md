# API

`POST /ask`, JSON in and out.

Request: `{"question": "<1 to 500 characters after trimming>"}`

Response 200:

```json
{
  "status": "answered",
  "segments": [
    {"text": "Multiplication takes precedence over addition.", "citations": [{"passage": 1, "cited_text": "each operator is evaluated in order of precedence"}]},
    {"text": " Parentheses override that order.", "citations": []}
  ],
  "passages": [
    {"n": 1, "chunk_id": "section-800:412", "book_title": "Introduction to Python Programming",
     "section_title": "1.5 Number basics", "page_start": 29, "page_end": 30, "text": "<full chunk text>"}
  ],
  "meta": {"model": "claude-opus-5-5", "setup": "section-800__pinecone-e5", "latency_ms": 4210}
}
```

- `status` is `answered`, `no_answer` (the documents do not contain the answer), or `refused` (Claude's `stop_reason` is `refusal` after fallbacks).
- For `no_answer`, `segments` holds one segment with the fixed sentence `The documentation does not answer this question.` and no citations. For `refused`, `segments` is empty.
- `passages` always lists the retrieved chunks in rank order, `n` starting at 1. `citations[].passage` refers to `n`.
- Concatenating `segments[].text` gives the full answer.

Errors: `{"error": {"code": "<code>", "message": "<one sentence>"}}` with status 400 `invalid_question`, 429 `daily_limit`, 502 `upstream_error`, 504 `timeout`.

## Local server

`python -m docent serve` listens on `localhost:8787` and allows CORS from `http://localhost:5173`. `serve --fake` answers from `web/src/fixtures/` and never calls Claude. It picks a fixture from the question text (`no_answer`, `refused`, `daily_limit`), and returns `answered` otherwise. `serve --fake NAME` returns that fixture for every question.
