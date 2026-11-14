# AWS Prerequisites and Configuration

None of these are verified yet. This is a checklist of what is missing or unconfirmed.

## Account and access
- [ ] An AWS account (or hackathon-provided sandbox) that all four members, or at least the member running the live calls, can use. **Unknown: who provides it?**
- [ ] **Bedrock model access** granted in the target region for `Amazon Titan Text Embeddings V2` and `Amazon Nova Lite` (Bedrock console → Model access). Some regions or accounts need it requested explicitly.
- [ ] Decide the **region** (e.g. `us-east-1` or `us-west-2`) and confirm that both models are available there.
- [ ] Check whether Nova Lite must be called through a **cross-region inference profile** (e.g. `us.amazon.nova-lite-v1:0`) rather than the bare model ID. This is account- and region-dependent, so test it in the smoke script.
- [ ] Check **quotas** (requests/min and tokens/min) for Titan and Nova Lite. Ingesting a large corpus makes many embedding calls, so size the concurrency accordingly.

## Credentials (never hardcoded)
- [ ] Choose one mechanism: **AWS SSO profile** (preferred), an IAM user with a profile in `~/.aws/credentials`, or environment variables, or an IAM role if deployed on AWS.
- [ ] Install and configure the AWS CLI (`aws configure sso` / `aws sso login --profile <name>`) on each developer machine.
- [ ] Use a **least-privilege IAM policy**: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` (Converse requires `bedrock:InvokeModel`) on the two model ARNs (plus the inference profile ARN if used). No other permissions are needed for this design.
- [ ] Confirm the team's rules on sharing credentials. Shared keys must never enter the repo, chat, or logs.

## Local configuration
- [ ] `.env.example` committed with variable names only. Real `.env` is git-ignored. Variables are listed in `api_contracts.md` §9.
- [ ] `.gitignore` covers `.env`, `data/chroma/`, `*.pem`, and AWS config.
- [ ] Python version pinned (3.11 or 3.12 suggested; confirm that the team's machines and the Chroma/LangChain wheels support it).
- [ ] Dependencies pinned: `boto3`, `fastapi`, `uvicorn`, `langchain` and the `langchain-aws` / `langchain-chroma` integrations (as needed), `chromadb`, `streamlit`, `pydantic-settings`, parsers (`pypdf` or `pymupdf`, `beautifulsoup4`, `markdown-it-py`), `rank-bm25` (if chosen), `pytest`.
- [ ] Windows note: the dev machine is Windows. Confirm that Chroma's native dependencies install cleanly (they may need Microsoft C++ Build Tools in some cases). `pdftotext` is available locally but must not be a runtime dependency.

## M0 status (2026-11-14)
- Implemented: env-driven config, `.env.example`, a lazy boto3 Bedrock Runtime client (default chain, optional `AWS_PROFILE`), and categorized error messages (credentials, IAM, model access, model ID/inference profile, region, throttling). Verified only with mocked boto3 unit tests.
- **Not verified:** no live AWS call has been made. Everything marked "unconfirmed" above stays open until `python scripts/test_bedrock.py --all` passes. Setup steps are in `README.md`.
- Python 3.12 venv is used locally (system default is 3.14).
- Chroma on Windows: `chromadb` 1.5.9 installed and ran (persistent store, tests) on Windows 11 with Python 3.12, with no C++ build tools needed.
- M1: the embedder sends one Titan request per chunk (`EMBEDDING_BATCH_SIZE` in parallel), so the Titan requests/minute quota matters for large corpora. Throttling is retried with backoff. The quota is still unchecked.

## Live verification log (2026-11-14, this machine)
- No profile selected: boto3 found no credentials. Set `AWS_PROFILE=hcl-rag` (an `aws login` profile; a local `.env` now does this).
- That profile type needs `botocore[crt]` (added to `requirements.txt`).
- Credentials then resolved. Both `amazon.titan-embed-text-v2:0` and `amazon.nova-lite-v1:0` calls in `us-east-1` were refused by AWS: *"Your account is currently being verified. Verification normally takes less than 2 hours"*. This is an account-level hold, not an IAM or model-access error. **Result: embed = FAIL, llm = FAIL (not verified).**
- Next: retry `python scripts/test_bedrock.py --all` after verification; then confirm Titan V2 and Nova Lite model access in the Bedrock console and whether Nova needs the `us.` inference profile.

## Live verification result (later on 2026-11-14)
After the account verification hold cleared, `python scripts/test_bedrock.py --all` **passed**: Titan V2 (`amazon.titan-embed-text-v2:0`, dimension 1024) and Nova Lite via the regional inference profile `apac.amazon.nova-lite-v1:0` (Converse API), region `ap-south-1`, profile `hcl-rag` (`aws login`, needs `botocore[crt]`). The `apac.` profile prefix matches the Asia Pacific region; use `us.` / `eu.` in those regions. Confirmed again with the application: ingestion (Titan), queries (Titan + Nova), a live evaluation, and a full workflow run.

## Open questions for the team
1. Single shared AWS account or per-member? Cost cap or budget alert?
2. Embedding dimension: 1024 (proposed) vs 512/256.
3. Is there a supplied document corpus, or do we synthesize one? Are there size limits?
4. Will the demo run only locally, or must it be deployed (affects credential handling)?
5. Does the brief's "link or identifier" for citations need clickable links? If so, the doc viewer is needed. The default is an identifier of the form `filename#section/p.N` plus `chunk_id`.
