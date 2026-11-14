# Enterprise Documentation Q&A Assistant

HCLTech Future Ready AI Engineer Hackathon, use case 2. Upload PDF, HTML, Markdown or TXT documents, ask questions in natural language, and get answers that use **only** the retrieved evidence, with validated citations, visible conflicts between sources, and a clear "not enough evidence" reply when the documents do not support an answer.

> **Status.** 461 automated tests pass offline (Bedrock mocked). The live path has also been verified: Titan Text Embeddings V2 and Nova Lite (`apac.amazon.nova-lite-v1:0`, region `ap-south-1`) succeed in `python scripts/test_bedrock.py --all`, and a full upload, question, citation, conflict and deletion workflow was run against real Bedrock. A live evaluation on 28 labelled questions is in `reports/evaluation_report.md`. Known weak spot: **chunk size** (see [Tuning chunk size](#tuning-chunk-size)).

## Quick start (Windows 11, two terminals)

```powershell
cd path\to\HCl-Hackathon
.\.venv\Scripts\Activate.ps1                       # first time only: see "Setup" below
```

**Terminal 1: FastAPI backend** (http://localhost:8000, API docs at http://localhost:8000/docs)

```powershell
uvicorn app.api.main:app --port 8000
```

**Terminal 2: Streamlit dashboard** (http://localhost:8501)

```powershell
streamlit run frontend/app.py --server.address localhost --server.port 8501
```

Open **http://localhost:8501** in your browser (Streamlit opens it automatically).

Helper scripts do the same without activating the venv (they call `.venv\Scripts\python.exe`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1      # Terminal 1
powershell -ExecutionPolicy Bypass -File scripts\start_ui.ps1       # Terminal 2
powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1      # optional: both, in two windows
powershell -ExecutionPolicy Bypass -File scripts\stop_all.ps1       # stop whatever listens on 8000 and 8501
```
All scripts accept ports (`-Port`, `-ApiPort`, `-UiPort`) and bind to localhost only.

**Using the dashboard**
1. **Documents** page: choose PDF/HTML/Markdown/TXT files (several at once), optionally enter a version label, click **Ingest documents**. Each file reports success or the exact reason it failed. The list below shows indexed documents with type, chunks and version, with **Refresh** and **Delete**.
2. **Ask Documents** page: type a question in the chat box. The answer shows numbered citations; expand **[1]**, **[2]**... for the verbatim excerpt, section, page and version, and **Retrieved evidence** for the ranking. A yellow box means the sources may disagree (both sides are shown); "Not enough evidence" means the documents do not support an answer. The sidebar can limit a question to specific documents.
3. **Evaluation** page: the real metrics, flagged cases and the full report. **System Status**: backend, ChromaDB, model configuration.

The synthetic demo corpus is in `data/eval/corpus/`. Upload `employee_handbook_v1.md` with version `1.0` and `employee_handbook_v2.md` with `2.0` (two separate uploads) to see conflict handling.

## 1. Overview and architecture

```
Streamlit UI  ->  FastAPI  ->  Titan embeddings (Bedrock)  ->  ChromaDB (persistent)
                      |                 vector search + BM25 -> Reciprocal Rank Fusion
                      |                 -> evidence strength -> conflict detection
                      +--------------> Nova Lite (Bedrock Converse) -> validated citations -> answer
```

The Streamlit app never calls Bedrock or ChromaDB; it talks only to FastAPI.

| Stage | Module | What it does |
|---|---|---|
| Parse | `app/ingestion/parsers/` | PDF (PyMuPDF), HTML (BeautifulSoup), Markdown, TXT into one block representation |
| Prepare | `app/ingestion/preprocess.py`, `chunking.py` | conservative cleaning; section/page-aware chunks with a `Document:`/`Section:` header |
| Embed | `app/embeddings/titan.py` | Titan V2 (1024 dims) for documents **and** queries |
| Store | `app/vectorstore/chroma_store.py` | persistent ChromaDB, cosine space, refuses a mismatched model/dimension |
| Retrieve | `app/retrieval/` | vector + BM25, Reciprocal Rank Fusion (k=60), evidence strength, conflict detection |
| Generate | `app/generation/` | Nova Lite via Converse, JSON output, programmatic citations, fallbacks |
| Serve | `app/api/` | FastAPI, lazy dependency container, uniform error body |
| UI | `frontend/` | Streamlit multi-page dashboard (`frontend/app.py` is the entry point) |
| Evaluate | `app/evaluation/` | labelled synthetic set, retrieval/answer/citation metrics, report |

Rules baked into the design: Titan for both documents and queries; ChromaDB (no Bedrock Knowledge Bases); credentials only from boto3's default chain; modules talk through the Pydantic models in `docs/api_contracts.md`.

**Behaviour you can rely on**
* Every citation is built from the real retrieved chunk (document, section, page, version, verbatim excerpt). Citations the model invents are removed; an answer left with none is withheld.
* With no usable evidence the LLM is not called at all.
* Conflicts that involve the cited passages are shown; the system never decides which version is current. Conflict detection is heuristic (numbers with units, dates, document "families").
* Answers never contain internal markers; refusals say what is missing.

## 2. Setup (first time)

```powershell
cd path\to\HCl-Hackathon
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env           # then edit values
```
`boto3[crt]` is required for profiles created with `aws login`.

### AWS authentication
**Never put keys in code or `.env`.** The app reads credentials through boto3's default chain; `.env` holds only a profile *name*.

```powershell
aws login --profile hcl-rag                 # or: aws configure sso --profile <name>; aws sso login --profile <name>
# in .env:  AWS_PROFILE=hcl-rag   AWS_REGION=ap-south-1   BEDROCK_LLM_MODEL_ID=apac.amazon.nova-lite-v1:0
python scripts/test_bedrock.py --all        # makes 2 tiny billable calls; exit code 0 only if both pass
```
A successful AWS CLI login does not prove Python can call Bedrock, which is why the smoke test exists. Nova Lite may require a regional inference profile (`apac.`, `us.`, `eu.` prefix) instead of the bare model ID; the profile must match your region. If AWS answers *"Your account is currently being verified"* the account is on a new-account hold: wait and retry (this happened earlier in this project and has since cleared).

### Environment configuration
Copy `.env.example` to `.env` (git-ignored). Invalid values raise an error naming the variable.

| Variable | Default | Notes |
|---|---|---|
| `AWS_REGION`, `AWS_PROFILE` | `us-east-1`, unset | profile name only |
| `BEDROCK_EMBED_MODEL_ID`, `BEDROCK_EMBED_DIM` | `amazon.titan-embed-text-v2:0`, `1024` | dimension 256/512/1024; fixed once indexed |
| `BEDROCK_LLM_MODEL_ID` | `amazon.nova-lite-v1:0` | or an inference profile ID |
| `CHROMA_PATH`, `CHROMA_COLLECTION` | `data/chroma`, `enterprise_docs` | |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `1000`, `150` | characters; see tuning note below |
| `MAX_UPLOAD_SIZE_MB`, `SUPPORTED_FILE_TYPES`, `EMBEDDING_BATCH_SIZE` | `25`, `pdf,html,htm,md,markdown,txt`, `8` | |
| `RETRIEVAL_TOP_N`, `MIN_RELEVANCE`, `RRF_K` | `20`, `0.30`, `60` | `MIN_RELEVANCE` is a heuristic cut-off, not a probability |
| `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, `MAX_CONTEXT_CHUNKS` | `800`, `0.0`, `6` | |
| `REPORTS_DIR`, `API_BASE_URL` | `reports`, `http://localhost:8000` | |

### API routes
`GET /health`, `POST /ingest`, `GET /documents`, `DELETE /documents/{doc_id}`, `POST /query` (grounded answer), `POST /retrieve` (evidence only, for debugging), `GET /evaluation/latest`, `GET /evaluation/report`. Errors are `{"error": {"code", "message"}}`; Bedrock failures are HTTP 502 with the AWS reason, never a stack trace.

```powershell
curl.exe -F "files=@C:\path\guide.pdf" -F "files=@C:\path\faq.md" -F "version=2.1" http://localhost:8000/ingest
curl.exe -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"How many days of annual leave do employees get?\", \"debug\": true}"
```
`version` is only what you supply; versions are never guessed. Documents are identified by **filename** (re-uploading a file with the same name replaces it; unchanged files are skipped). Response statuses: `answered`, `conflict`, `needs_clarification`, `not_found`.

### Tuning chunk size
The default `CHUNK_SIZE=1000` merges short unrelated paragraphs of a document with no headings into one chunk, which blurs its embedding. On the synthetic corpus, 400-character chunks gave status accuracy 0.964 and 1 to 2 real failures; the 1000 default gave 0.857 and 4 false "not enough evidence" replies (all from one 5-rule IT policy file). One small synthetic corpus does not prove a better default for your documents. To experiment: set `CHUNK_SIZE=500` and `CHUNK_OVERLAP=75` in `.env`, restart the API, delete and re-upload the documents (the setting only applies at ingestion), and check with `python scripts/run_evaluation.py --live --chunk-size 500 --chunk-overlap 75`.

## Deploying (Render backend + Streamlit Cloud frontend)

The dashboard needs a reachable backend. If a deployed dashboard says **Backend offline**, its `API_BASE_URL` is still `http://localhost:8000` (the Streamlit Cloud server itself). Set `API_BASE_URL` (and `API_KEY`) in Streamlit Cloud -> Secrets, and host the FastAPI backend on a public https URL. Full, step-by-step instructions (Render blueprint `render.yaml`, environment variables, secure AWS credentials, ChromaDB persistence on a disk, testing `/health`, Streamlit secrets) are in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**. Local development is unchanged. Set `API_KEY` on a public backend, otherwise anyone with the URL can upload documents and spend your Bedrock budget.

## 3. Run the tests

```powershell
python -m pytest                                    # 461 offline tests; no AWS, no network
python -m pytest tests/test_generation.py           # one file
python -m pytest tests/test_chroma_store.py::test_persists_across_instances   # one test
```

## 4. Run the evaluation

```powershell
python scripts/run_evaluation.py --live      # real Titan + Nova Lite (billable: about 50 Titan + 28 Nova calls)
python scripts/run_evaluation.py             # offline dry run: stand-in embedder, no answers; NOT representative
python scripts/run_evaluation.py --live --baseline reports/eval_results_run1_original.json --analysis data/eval/run1_analysis.md
python scripts/run_evaluation.py --live --chunk-size 1000 --chunk-overlap 150 --out reports/deployed_defaults
```
Outputs `reports/evaluation_report.md` and `reports/eval_results.json` (shown on the dashboard's Evaluation page). The corpus is indexed into a temporary store, so your own collection is never touched. Answer metrics are **automatic proxies, not human judgements**; the report contains a blank human-review table.

**Current live results** (28 questions, 8 documents, 400-character chunks; details, per-case analysis and the before/after comparison are in the report):

| Metric | Result |
|---|---|
| Recall@5 / MRR / nDCG@5 | 1.000 / 0.976 / 0.982 |
| Failed queries | 0 of 28 |
| Status accuracy / key-fact recall | 0.964 / 0.975 |
| Citation validity / precision / recall | 1.000 / 1.000 / 1.000 |
| Conflicts surfaced (4 conflicting-version questions) | 4 of 4 |
| Flagged entries | 6 on 6 questions: 2 failures (q11 false abstention, q12 intermittent incompleteness), 4 informational notes |

The original run reported "15 failure cases". That counted **entries**, not failed queries (0 failed): they fell on 11 questions and mixed real defects (conflict status forced by unrelated passages, wrongly withheld answer, a stop-word bug, "latest version" wording) with metric artifacts and handled abstentions. See sections 9 to 11 of the report. The first run is preserved in `reports/*_run1_original.*`, later runs in `reports/history/`.

## 5. Project structure

```
app/            api/ core/ embeddings/ vectorstore/ ingestion/ retrieval/ generation/ evaluation/ schemas/
frontend/       app.py (entry point), pages/ (Documents, Ask Documents, Evaluation, System Status), components/, api_client.py
scripts/        start_api.ps1 start_ui.ps1 start_all.ps1 stop_all.ps1, test_bedrock.py, run_evaluation.py, build_eval_corpus.py
data/eval/      synthetic corpus + labelled questions + analysis
reports/        evaluation_report.md, eval_results.json (+ run history, deployed_defaults/)
docs/           architecture, API contracts, implementation plan, AWS prerequisites, PROGRESS
tests/          461 offline tests
```

## 6. Known limitations

* **Chunk size / diluted chunks:** short, unrelated paragraphs merged into one chunk can be falsely refused (see tuning note). `MIN_RELEVANCE` (0.30) and the evidence margins are heuristics; one live question (q11, a paraphrase against a diluted chunk) is still falsely refused with 400-character chunks.
* **LLM variation:** Nova's wording and willingness to answer from weakly matching text vary slightly between runs even at temperature 0; single-question differences between evaluation runs are noise.
* **Conflict detection** is heuristic: it compares numbers with units and dates between passages with the same section name, and document "families". It misses negation and paraphrase, and can raise false alarms.
* No OCR (scanned PDFs are rejected with a clear message); PDF tables are not reconstructed; PDF headings are a font-size heuristic.
* Document identity is by filename; versions are only what you supply.
* Titan is called once per chunk, so large corpora are slow and may be throttled.
* The BM25 index is cached per process; writes by another process to the same Chroma directory are not noticed until restart.
* The evaluation corpus and questions are small, synthetic and team-written; automatic answer metrics are lexical proxies, and the groundedness proxy was made more lenient than its first version on purpose (both values are reported).
* Chat history is per browser session. A real browser was not driven in this verification: the dashboard was verified by HTTP checks and by running every page through Streamlit's test harness against the live backend (including upload, ask, delete).
