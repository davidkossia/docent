# Docent infrastructure

One CDK stack, `DocentStack`, serves `POST /ask` from AWS Lambda behind an API Gateway HTTP API.

| Resource | Setting |
|---|---|
| Lambda | Python 3.12, arm64, 1024 MB, 29 s timeout, reserved concurrency 2 |
| HTTP API | One route, `POST /ask`; stage throttle rate 2 per second, burst 5; CORS for one origin |
| DynamoDB | On-demand table that counts questions per UTC day; items expire after 3 days |
| Keys | SSM Parameter Store SecureString parameters, read by the Lambda at cold start |
| Logs | CloudWatch, 2-week retention; status, latency, and token counts only |

Each request runs in this order: validate the question, count it against the daily limit, then
search Pinecone and call Claude. A request over the limit gets `429 daily_limit` and never reaches
Pinecone or Claude.

## Deploy

You need AWS credentials for the target account, Node.js 22, and `uv`.

1. Store the two keys. Use the default `aws/ssm` key (no `--key-id`); the Lambda role then needs no
   KMS permission. `read -rs` reads each key without echoing it or saving it in your shell history.
   `printf` is a shell built-in, and `--value file:///dev/stdin` makes the AWS CLI read the key from
   the pipe, so the key never appears in a process's arguments.

   ```bash
   read -rs KEY && printf '%s' "$KEY" | aws ssm put-parameter --region us-east-1 \
     --name /docent/anthropic_api_key --type SecureString --value file:///dev/stdin
   read -rs KEY && printf '%s' "$KEY" | aws ssm put-parameter --region us-east-1 \
     --name /docent/pinecone_api_key --type SecureString --value file:///dev/stdin
   unset KEY
   ```

2. Check that the account can reserve 2 concurrent executions. AWS keeps 10 unreserved, so
   `ConcurrentExecutions` must be 12 or more. New accounts often start at 10.

   ```bash
   aws lambda get-account-settings --region us-east-1 --query AccountLimit.ConcurrentExecutions
   ```

3. Build the Lambda bundle. The script installs only the hash-locked packages in
   `infra/scripts/requirements-lambda.lock`. It fails if `backend/docent` has uncommitted changes or
   if the unzipped bundle passes 200 MB. After you change `backend/requirements-lambda.txt`, run
   `infra/scripts/lock_lambda.sh` and commit the new lock.

   ```bash
   infra/scripts/build_lambda.sh
   ```

4. Deploy. The first deploy in an account and region needs `npx cdk bootstrap` once.

   ```bash
   cd infra
   npm ci
   npx cdk deploy -c allowedOrigin=https://<the UI's origin> -c setup=<strategy>__pinecone-e5
   ```

   `allowedOrigin` has no default, so `cdk synth`, `cdk deploy` and `cdk destroy` fail without it.
   It must use `https`; `http` is allowed only for `localhost`, `127.0.0.1`, and `[::1]`.

   The stack prints `ApiUrl`. Set the UI's `VITE_API_URL` to it.

Context values and their defaults (in `cdk.json`, except `allowedOrigin`):

| Key | Default | Meaning |
|---|---|---|
| `region` | `us-east-1` | Same region as the Pinecone index and the SSM parameters |
| `allowedOrigin` | none (required) | The one origin CORS allows; `*` is rejected |
| `dailyLimit` | `150` | Questions per UTC day that may reach Claude |
| `setup` | `section-800__pinecone-e5` | Pinecone namespace to search; the embedder must be Pinecone-hosted |

To change the daily limit without a deploy, edit the function's `DAILY_LIMIT` variable in the
Lambda console. Set it to `0` to stop all Claude calls at once.

## Roll back

To remove everything the stack created, run:

```bash
cd infra && npx cdk destroy -c allowedOrigin=https://<the UI's origin>
```

`cdk destroy` leaves the two SSM parameters, because you created them. To delete them, run
`aws ssm delete-parameter --region us-east-1 --name <name>` for each.

## Worst-case daily cost at the default limit

Claude calls are almost all of the cost, and the daily limit bounds them.

**Claude.** Each counted question makes at most one request (`max_retries=0`). Per attempt,
`max_tokens` is 16,000, and thinking tokens count against it. Input is at most about 5,000 tokens:
five passages (about 7,000 characters with section-1200 chunks), the system prompt, and a question
of at most 500 characters.

| Case | Per question | 150 questions |
|---|---|---|
| One attempt on Claude Opus 5.5 ($4 input, $20 output per million tokens) | 5,000 × $4/1M + 16,000 × $20/1M = $0.020 + $0.320 = $0.340 | $51.00 |
| Opus 5.5 refuses after full output, then the server-side fallback runs on Claude Opus 5 or Opus 4.8 ($5, $25), today's fallback targets | $0.340 + (5,000 × $5/1M + 16,000 × $25/1M) = $0.340 + $0.425 = $0.765 | $114.75 |

The second row is the ceiling. It needs every question to use all 16,000 output tokens, refuse
after that output, and then use all 16,000 again on the fallback model. Typical answers use a small
fraction of `max_tokens`. The fallback figure assumes today's fallback targets for Claude Opus 5.5
(Claude Opus 5 and Claude Opus 4.8). If Anthropic changes those targets or their prices, recompute it.

**AWS.** The stage throttle allows at most 2 requests per second, or 172,800 a day. At that rate,
API Gateway, Lambda, DynamoDB writes (a refused conditional write is still billed), and CloudWatch
Logs together cost under $1 a day at us-east-1 list prices.

**Pinecone.** Search runs only after the quota check, so at most 150 queries and 150 query
embeddings a day.

**Total ceiling:** about $116 a UTC day; about $52 a UTC day when no question falls back.

The counter resets at midnight UTC, so any 24-hour window that spans midnight can hold two days'
limits: 300 questions, or up to $229.50 ($102.00 without fallbacks) plus AWS.

| Period | Questions | Ceiling | Without fallbacks |
|---|---|---|---|
| One UTC day | 150 | $114.75 + under $1 AWS | $51.00 + under $1 AWS |
| Any 24 hours across midnight UTC | 300 | $229.50 + under $1 AWS | $102.00 + under $1 AWS |
| A 31-day month | 4,650 | 31 × $114.75 = $3,557.25 + under $31 AWS | 31 × $51.00 = $1,581.00 + under $31 AWS |

Set a monthly spend limit on the Anthropic workspace as the backstop.

**One client can use the whole day's limit.** At the stage throttle of 2 requests per second, one
client can send 150 questions in 75 seconds; everyone else then gets `429 daily_limit` until
midnight UTC. HTTP APIs have no per-client limit. A per-IP rate rule needs AWS WAF in front of the
API (a later change).

## Slow Claude calls

API Gateway HTTP APIs end every request at 30 seconds. The stack keeps the whole request inside
that limit:

1. The Lambda timeout is 29 seconds.
2. The handler gives the Claude request a client timeout equal to the time the Lambda has left,
   minus 2 seconds, with no retries.
3. When that timeout passes, the Anthropic SDK raises `APITimeoutError`. `service.ask` maps it to
   `504 timeout`, and the browser gets the contract's error body.
4. A wall-clock alarm 0.5 seconds later, repeating every 0.5 seconds until the request returns, is
   a backstop for a network read that the client timeout does not bound. A hung Pinecone search
   that the alarm stops returns `502 upstream_error`.

A timed-out question still counts against the daily limit, because Claude may have billed it.
