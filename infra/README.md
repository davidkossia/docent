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
   KMS permission. `read -rs` keeps each key out of your shell history.

   ```bash
   read -rs ANTHROPIC_KEY && aws ssm put-parameter --region us-east-1 \
     --name /docent/anthropic_api_key --type SecureString --value "$ANTHROPIC_KEY"
   read -rs PINECONE_KEY && aws ssm put-parameter --region us-east-1 \
     --name /docent/pinecone_api_key --type SecureString --value "$PINECONE_KEY"
   unset ANTHROPIC_KEY PINECONE_KEY
   ```

2. Check that the account can reserve 2 concurrent executions. AWS keeps 10 unreserved, so
   `ConcurrentExecutions` must be 12 or more. New accounts often start at 10.

   ```bash
   aws lambda get-account-settings --region us-east-1 --query AccountLimit.ConcurrentExecutions
   ```

3. Build the Lambda bundle. The script fails if the unzipped bundle passes 200 MB.

   ```bash
   infra/scripts/build_lambda.sh
   ```

4. Deploy. The first deploy in an account and region needs `npx cdk bootstrap` once.

   ```bash
   cd infra
   npm ci
   npx cdk deploy -c allowedOrigin=https://<the UI's origin> -c setup=<strategy>__pinecone-e5
   ```

   The stack prints `ApiUrl`. Set the UI's `VITE_API_URL` to it.

Context values and their defaults (in `cdk.json`):

| Key | Default | Meaning |
|---|---|---|
| `region` | `us-east-1` | Same region as the Pinecone index and the SSM parameters |
| `allowedOrigin` | `http://localhost:5173` | The one origin CORS allows; `*` is rejected |
| `dailyLimit` | `150` | Questions per UTC day that may reach Claude |
| `setup` | `section-800__pinecone-e5` | Pinecone namespace to search; the embedder must be Pinecone-hosted |

To change the daily limit without a deploy, edit the function's `DAILY_LIMIT` variable in the
Lambda console. Set it to `0` to stop all Claude calls at once.

## Roll back

To remove everything the stack created, run:

```bash
cd infra && npx cdk destroy
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
| Opus 5.5 refuses after full output, then the server-side fallback runs on Claude Opus 5 or Opus 4.8 ($5, $25) | $0.340 + (5,000 × $5/1M + 16,000 × $25/1M) = $0.340 + $0.425 = $0.765 | $114.75 |

The second row is the ceiling. It needs every question to use all 16,000 output tokens, refuse
after that output, and then use all 16,000 again on the fallback model. Typical answers use a small
fraction of `max_tokens`.

**AWS.** The stage throttle allows at most 2 requests per second, or 172,800 a day. At that rate,
API Gateway, Lambda, DynamoDB writes (a refused conditional write is still billed), and CloudWatch
Logs together cost under $1 a day at us-east-1 list prices.

**Pinecone.** Search runs only after the quota check, so at most 150 queries and 150 query
embeddings a day.

**Total ceiling:** about $116 a day; about $52 a day when no question falls back. Set a monthly spend
limit on the Anthropic workspace as the backstop.

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
