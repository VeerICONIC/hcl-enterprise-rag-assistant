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

## Open questions for the team
1. Single shared AWS account or per-member? Cost cap or budget alert?
2. Embedding dimension: 1024 (proposed) vs 512/256.
3. Is there a supplied document corpus, or do we synthesize one? Are there size limits?
4. Will the demo run only locally, or must it be deployed (affects credential handling)?
5. Does the brief's "link or identifier" for citations need clickable links? If so, the doc viewer is needed. The default is an identifier of the form `filename#section/p.N` plus `chunk_id`.
