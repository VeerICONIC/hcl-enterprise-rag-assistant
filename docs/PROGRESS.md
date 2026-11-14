# Project Progress

Living document. **Read at the start of every Claude Code session. Update before ending one.** Keep entries short and factual.

## Current state
- **Phase:** M0 to M6 implemented **and verified live.** 431 offline tests pass. Live AWS works (Titan V2 + Nova Lite via `apac.amazon.nova-lite-v1:0`, region `ap-south-1`): the smoke test passes, the full upload -> question -> citation -> conflict -> delete workflow was run against real Bedrock, and a live evaluation exists (`reports/evaluation_report.md`).
- **Last updated:** 2026-11-14 (evaluation-analysis session)
- **Awaiting:** the user's decisions listed under "Decisions needing approval (evaluation-analysis session)"; chiefly the chunk-size default.

## Evaluation analysis + live verification (2026-11-14, evaluation-analysis session)
**Context.** The user ran the live smoke test and the first live evaluation (mode live, `apac.amazon.nova-lite-v1:0`, `ap-south-1`; reports/eval_results.json). This session re-confirmed AWS access (`test_bedrock.py --all`: embed PASS, llm PASS), preserved that run unchanged (`reports/*_run1_original.*`), analysed all 15 flagged entries, fixed the real defects, re-ran the live evaluation, and verified the application end to end.

**What "failure cases: 15" meant.** 15 *entries* on 11 distinct questions; 0 of 28 queries failed. Breakdown (details in `reports/evaluation_report.md` sections 9 and 10, `data/eval/run1_analysis.md`): 6 groundedness-proxy flags (5 pure metric artifacts: the proxy penalised words copied from the question or conflict framing; 1 also a real wording defect), 4 wrong-status (q01 generator forced `conflict`; q16 answer wrongly withheld; q26 tokenizer bug; q11 evidence gate), 3 informational abstention notes (answer stage correctly declined), 1 false conflict (q01), 1 completeness gap (q12). No evaluation label was wrong; none was changed.

**Real defects found and fixed (code):**
- Tokenizer: personal pronouns (my, me, ...) were not stop words, which broke the lexical-rescue rule for "How do I reset my password?".
- Generator: a conflict now only counts if it involves a passage the answer cites (previously any retrieved conflict forced `conflict`; q01).
- Generator: Nova's own `conflict` status is accepted only if the cited passages actually differ (found live: identical sick-leave text in two versions was reported as a conflict).
- Generator: lenient evidence-alias normalization; an unusable claim list no longer discards an answer with valid inline citations (q16).
- Generator/prompt: the model may not call a version latest/newer/older/previous/current unless a source says so, plus a deterministic warning (q18); a partial-answer rule (answer what the evidence supports, refuse only if nothing answers); the weak-evidence note no longer contradicts it; the prompt tells the model the automatic conflict notes may be unrelated.
- Generator: for `weak` evidence every returned passage is shown to the model (a correct passage with low cosine could rank below others and never reach the prompt; found live with default chunking).
- Refusals/clarifications no longer leak internal `[E#]` aliases (found live); a sentence-initial alias becomes "A retrieved passage".
- Evaluation reporting: failure entries now have a severity (failure / review / note) and counts, so the headline no longer mixes defects with proxy flags.

**Methodology changes (documented in the report; original run preserved):** groundedness proxy v2 (question-copied and conflict-framing words ignored; numbers must appear in the cited text or its version/file metadata; checkable = has a citation marker or a number; word coverage judged only with 3+ content words). The run-1 proxy is still reported as `groundedness_v1` on the same answers; **v2 is more lenient by design, so the rise 0.74 -> 1.0 is partly measurement** (like-for-like v1: 0.74 -> about 0.83-0.90 across runs). `false_conflict` measures the conflict shown to the user; retrieval-only flags are the note `retrieval_conflict_not_shown`. No threshold (`MIN_RELEVANCE`, lexical-rescue coverage, evidence margins) and no label was changed to improve scores.

**Before/after (run 1 -> final, live, same questions, 400/60 chunks):** Recall@5 1.000 -> 1.000, MRR 0.976 -> 0.976, nDCG 0.982 -> 0.982; status accuracy 0.857 -> 0.964; conflicts surfaced 0.75 -> 1.0; key-fact recall 0.972 -> 0.975; citation validity/precision/recall 1.0 throughout; questions with a real `failure` entry 5 -> 2 (q11, q12); flagged entries 15 -> 6. Intermediate runs are in `reports/history/`.

**Chunk-size finding (open decision).** Evaluated at the deployed default (1000/150, `reports/deployed_defaults/`): status accuracy 0.857 and 4 false "not enough evidence" replies (q05, q07, q11, q26), all from the 5-rule IT policy file, which becomes a single diluted chunk. Recall@5 cannot see this (the chunk is retrieved; the evidence gate and the model then decline). The default was not changed.

**Application verification (live, isolated throwaway store, ports 8810/8811, since removed):** `scripts/start_all.ps1 -Hidden` started FastAPI and Streamlit; upload of all four formats (8 documents, 22 chunks) via the same `ApiClient` the pages use; unchanged re-upload skipped (0 new chunks); `.docx` rejected (400); ChromaDB persistence confirmed across an API restart (22 chunks, 8 documents); questions: direct answered with citations, 20 vs 25 annual-leave conflict shown with both sources and version labels, unsupported question refused without citations, document filter respected, deleting a document removed its chunks (22 -> 20) and its content could no longer be retrieved (LLM not called). Streamlit pages were driven with `streamlit.testing` against the live backend: Overview (real counts), Documents (list, delete, and multi-file upload of a PDF and a Markdown file), Ask (answer, citations, conflict warning, ranking panel; unsupported question warning), Evaluation (real metrics and report), System Status. **Not done:** a real browser was not opened or driven (HTTP 200 and Streamlit health only).

**Windows launchers:** `scripts/start_api.ps1`, `start_ui.ps1`, `start_all.ps1` (optional, `-Hidden`/`-NoBrowser` for testing), `stop_all.ps1`. Bugs found by running them: `localhost` can resolve to IPv6 in PowerShell 5.1 while uvicorn binds IPv4 (the wait loop uses 127.0.0.1); an array parameter passed through `-File` collapses (`-Ports` is a comma string).

**Tests: 431 passed** (`python -m pytest`, about 20 s). New this session: generator fixes, severity/proxy-v2, report endpoint, launch-script parse checks.

### Decisions needing approval (evaluation-analysis session)
1. **Chunk size:** keep `CHUNK_SIZE=1000` or change the default (data suggests ~400-500 for documents made of short unrelated rules; unproven on real documents).
2. Additive API: `GET /evaluation/report` (Markdown for the dashboard). New eval options `--chunk-size/--chunk-overlap/--baseline/--analysis`.
3. Groundedness proxy v2 and failure severity as the reporting method (v1 is still reported for comparison).
4. The generator now shows all returned passages when evidence is `weak`, and accepts the model's own `conflict` only if cited passages differ.
5. Evaluation intermediate files were moved to `reports/history/` (kept, not deleted).

### Known open issues
q11 (paraphrase vs a diluted chunk; cosine 0.269 < 0.30, no keyword overlap: false abstention, not tuned), q12 (intermittent incompleteness), the retrieval-level conflict flag among unrelated passages (hidden from users, recorded as a note), LLM run-to-run variance, the chunk-size default, and "How do I reset my password?" still being refused at the 1000-character default (Nova declines a correct but brief passage).

## M4 + M5 + M6 delivered (2026-11-14, master-prompt session)

**M4 frontend** (`frontend/`): `app.py` (Overview), `pages/1_Documents.py`, `2_Ask_Documents.py`, `3_Evaluation.py`, `4_System_Status.py`, `api_client.py` (httpx; the frontend's only data path), `config.py`, `components/` (`ui`, `sidebar`, `document_manager`, `chat_interface`, `source_viewer`, `conflict_viewer`, `metrics`). The UI never imports boto3 or chromadb (asserted by a test). It shows real backend numbers only, and errors (backend down, HTTP errors, AWS failures) render as messages, never tracebacks. Chat history is per browser session. Fixed while running it: the delete confirmation now survives the rerun, the deprecated `use_container_width` was replaced, and mixed-type table columns were made strings.
- New backend route: `GET /evaluation/latest` (read-only, serves `reports/eval_results.json`). New setting `REPORTS_DIR`.
- Launch: `streamlit run frontend/app.py --server.address localhost --server.port 8501` (the default binds to the network).

**M5 evaluation** (`app/evaluation/`): `dataset.py`, `retrieval_metrics.py` (Recall@k, MRR, nDCG@k, hit rate), `answer_metrics.py`, `citation_metrics.py`, `evaluator.py`, `report_generator.py`, `offline.py`; `scripts/run_evaluation.py`, `scripts/build_eval_corpus.py`. Synthetic fictional corpus (8 documents: PDF, HTML, MD, TXT; two conflicting handbook versions) and 28 labelled questions (direct 6, paraphrase 5, multi-document 3, conflicting-version 4, ambiguous 3, unanswerable 4, similar-keywords 3) in `data/eval/`.
- Labels are text anchors (`filename` + phrase) resolved to chunk IDs at run time, because chunk IDs are unstable (see decisions).
- The corpus is indexed into a **temporary** store; the user's collection is never touched (tested).
- **Offline mode** uses a hashed bag-of-words stand-in embedder and no LLM, and is labelled "NOT representative" everywhere. **Live mode** uses real Titan + Nova and has not been run.
- Measured offline (machinery only): Recall@5 0.952, MRR 0.873, nDCG@5 0.894 over 21 answerable questions; the conflict heuristic flagged 4/4 conflicting-version questions. Failure cases are listed in the report: a residual false conflict on q01 (the retrieved annual-leave chunks genuinely differ), one paraphrase miss, and weak-evidence results for some ambiguous/unanswerable questions. These say nothing about Titan/Nova.
- Detector refinements driven by the evaluation: value sets must be disjoint per unit; two chunks with named sections are the same subject only if the section names match; identical passages in two versions are not a version conflict.
- `reports/evaluation_report.md` and `reports/eval_results.json` hold the offline dry run, with 7 evidence examples, failure cases, limitations, recommendations and a blank human-review table.

**M6 integration:** README rewritten (all 18 required sections). A local `.env` was created (git-ignored; only `AWS_PROFILE=hcl-rag`). Requirements now include `streamlit`, `httpx` and `boto3[crt]`.
- **Tests: 402 passed** (`python -m pytest`, about 18s), including `tests/test_frontend.py` (ApiClient against the real FastAPI app; Streamlit `AppTest` for every page with a stub backend) and `tests/test_evaluation.py`.
- **Manual end-to-end check (no usable AWS):** FastAPI and Streamlit ran together; `/health`, `/evaluation/latest` and Streamlit's health endpoint responded; all five pages rendered against the live backend; `/query` and the Ask page returned the clear AWS error message.
- Blocked or not done: any live Titan/Nova call; the live evaluation; calibrating thresholds; testing with real documents.

### Decisions needing approval (master-prompt session)
1. `GET /evaluation/latest` and the `REPORTS_DIR` setting were added (additive). The contract lists an optional `POST /evaluate`, which was not built because live evaluation is billable and runs from the CLI.
2. Evaluation labels are text anchors, not `relevant_chunk_ids` (contract section 8); the resolved IDs are stored in the results.
3. `QueryResponse.latency_ms` keys are `retrieve` (includes the query embedding), `generate`, `total` (the contract lists `embed` too).
4. `POST /retrieve` is kept alongside `/query` as a debugging route.
5. The conflict heuristic now requires matching section names for two passages to count as the same subject (fewer false alarms, more misses).
6. A local `.env` with `AWS_PROFILE=hcl-rag` was created (profile name only; git-ignored).

## M2 hardening + M3 delivered (2026-11-14, master-prompt session)
**Verified first:** `ChromaStore.query()` was already implemented (M2); 293 tests passed at session start (the older "185" figure was stale).

**M2 hardening**
- BM25 index is now cached per metadata filter and invalidated by `ChromaStore.revision` (bumped by upsert, `delete_ids`, `delete_document`), so additions, deletions and re-ingests are reflected and unchanged collections are not rebuilt. Limitation: the counter is per process.
- Account-verification hold from AWS is reported explicitly (`PermissionDeniedError`, "AWS is still verifying this account").

**M3 (generation)** in `app/generation/`: `bedrock_llm.py` (`BedrockNovaClient`, Converse API, model ID or inference profile from `BEDROCK_LLM_MODEL_ID`), `prompts.py`, `citation_builder.py`, `response_validator.py`, `generator.py` (`AnswerGenerator`), `models.py`.
- Evidence is addressed by aliases (`E1`...), mapped back to real chunk IDs by us. Nova returns JSON (`status`, `answer` with `[E#]` markers, `claims`, `clarifying_question`).
- Citations are built **only** from retrieved chunk metadata. The excerpt is a verbatim substring of the chunk text (best-overlapping sentence). Unknown aliases are removed with a warning; claims without a valid citation are dropped and the answer rebuilt from supported claims; an answer with no valid citation is withheld as `not_found`.
- `evidence_strength == "none"` (or no chunks) returns `not_found` **without calling the LLM**. Weak evidence caps confidence at `low`. Retrieval conflicts force status `conflict` and travel in `Answer.conflicts`. Confidence is a heuristic label, not a probability.
- Malformed model JSON gets one repair retry, then `GenerationError` (HTTP 502 `generation_failed`). Bedrock errors propagate as 502 `bedrock_error`.
- `POST /query` (contract `QueryRequest` -> `QueryResponse`) via `app/api/query_service.py`. `latency_ms` keys: `retrieve` (includes the question embedding), `generate`, `total`. `debug=true` returns ranked chunks. `POST /retrieve` is kept for debugging. **This resolves M2 decision 1.**
- Config added: `LLM_MAX_TOKENS` (800), `LLM_TEMPERATURE` (0.0), `MAX_CONTEXT_CHUNKS` (6). Dependency change: `boto3[crt]` (needed for profiles created with `aws login`).
- Tests: **356 passed** (offline; mocked Nova). New: `tests/test_generation.py`, `tests/test_api_query.py` (+ BM25 cache and AWS-hold tests).

**Real AWS status (measured this session):** with no profile selected boto3 found no credentials. With `AWS_PROFILE=hcl-rag` (an `aws login` profile) boto3 needed `botocore[crt]` (now in requirements); after installing it the credentials resolved, but Bedrock refused both Titan and Nova calls with *"Your account is currently being verified"* (AWS account hold, region us-east-1). So **no live model call has succeeded**. Retry `AWS_PROFILE=hcl-rag python scripts/test_bedrock.py --all` after AWS finishes verifying the account.

## M2 delivered (2026-11-14)
- `ChromaStore.query(embedding, top_k, where)` returns `(Chunk, cosine_similarity)` best-first (`similarity = 1 - Chroma cosine distance`, clamped to [-1, 1]; ties by `chunk_id`; empty collection gives `[]`; `top_k` larger than the collection is fine; invalid, zero, non-finite or wrong-dimension embeddings and a bad `top_k` raise `VectorStoreError`; malformed stored records are skipped with a warning). Additive store methods: `get_embeddings(ids)`, and `get_all_chunks(where=None)` (now ordered by `chunk_id`).
- `app/retrieval/`: `retriever.py` (`Retriever`, `build_where`), `vector_retriever.py`, `bm25_retriever.py` (own Okapi BM25, no new dependency), `hybrid_ranker.py` (RRF), `evidence_analyzer.py`, `conflict_detector.py`, `models.py` (`RetrievalSettings`, internal dataclasses, `InvalidQueryError`), `text.py` (tokenizer).
- Query embedding: the existing `TitanEmbedder.embed_query` already used the same model, dimension and `BedrockRuntime` as documents, so no new embedding code was needed. It is now covered by retrieval tests.
- API: `POST /retrieve` (see decision 1). `Container.retriever` is built lazily. `InvalidQueryError` maps to 400 `invalid_query`.
- Config: one new setting, `RRF_K` (default 60). `RETRIEVAL_TOP_N` and `MIN_RELEVANCE` already existed and are now used.
- Dependencies added: none.
- Tests: **293 passed** in total (`python -m pytest`, about 8s, offline). 108 are new, covering the store query, BM25, RRF, evidence, conflicts, the retriever and the API.
- Manual check of the real server (no AWS): `/retrieve` without credentials returned 502 with the credentials message, a punctuation-only question returned 400 `invalid_query`, and an empty question with `top_k: 0` returned 422.
- Not done (out of scope): generation, citations, `/query`, frontend, evaluation, and calibrating thresholds on real data.

### M2 design decisions (need approval)
1. **Endpoint:** the contract's `POST /query` returns `QueryResponse`, whose `answer: Answer` is required, but M2 must not generate answers. Any placeholder `Answer` would misreport a status. I added **`POST /retrieve`** (body `QueryRequest`, response `RetrievalResult`, both existing contract types) and left `/query` for M3. No schema changed. If you prefer another route name, or a placeholder `/query`, it is a small change.
2. **`RetrievedChunk.score`** (contract: normalized 0..1) is the RRF sum divided by its theoretical maximum `2/(k+1)`. It measures rank agreement between vector and BM25, and is not a probability. `vector_score` is the true cosine.
3. **`RetrievalResult.top_score`** is undefined in the contract. I use the best cosine among the returned chunks (0.0 if none), because that is what `MIN_RELEVANCE` is compared to.
4. **Evidence strength** thresholds are provisional and chosen before any live Titan data: relevant means cosine >= `MIN_RELEVANCE` (0.30) or lexical rescue (>= 80% of a 2+-term query's content words); strong means best cosine >= 0.45 with 2+ relevant chunks, or >= 0.60 alone; otherwise weak. They need calibrating on the M5 evaluation set.
5. **Conflict encoding:** `ConflictGroup` has no confidence field, so `version_difference` means metadata/filename based and `contradiction`/`date_difference` mean text heuristic. The description starts with `METADATA CONFLICT` or `POSSIBLE` and says what is unknown. Version metadata is often missing, because M1 only records caller-supplied versions, so most version groups will be `POSSIBLE`. Detection is heuristic only and never picks a "current" version.
6. **Same-document pairs are never compared**, and only relevant chunks participate.
7. **BM25 rebuilds its index on every query** from `get_all_chunks`. This is fine for a small collection and a known limit for a large one.
8. **Returned chunks:** `chunks` is the fused top_k even when evidence is `none`, so M3 must check `evidence_strength` before using them.
9. **Punctuation-only questions** (no letters or digits) are rejected as 400 `invalid_query`.

### M2 known limitations
Heuristic conflict detection (numbers, units and dates only; English patterns; no negation or paraphrase understanding; false positives and misses expected). Uncalibrated thresholds. No stemming or synonyms in BM25. RRF treats both retrievers equally. Everything is validated only against mocked, hashed-bag-of-words embeddings, never real Titan vectors.

## M1 delivered (2026-11-14)
- `app/ingestion/`: `parsers/{base,pdf,html,markdown,text}_parser.py`, `parser_factory.py`, `preprocess.py`, `chunking.py`, `ids.py`, `errors.py`, `models.py` (internal `ParsedDocument`/`Block`/`ChunkDraft`), `ingestion_service.py` (`IngestionService`, independent of FastAPI).
- `app/embeddings/`: `base.py` (`Embedder` protocol, `EmbeddingError`), `titan.py` (`TitanEmbedder` over the M0 `BedrockRuntime`; throttle retries; dimension validation).
- `app/vectorstore/chroma_store.py`: `ChromaStore` (persistent, cosine, records embedding model and dimension, refuses a mismatched reopen). Implements `upsert`, `delete_document`, `delete_ids`, `count`, `list_documents`, `get_document_info`, `get_document_state`, `get_chunks`, `get_all_chunks` and `health`. **`query` is not implemented (M2).**
- `app/api/`: `main.py` (`GET /health`, `POST /ingest`, `GET /documents`, `DELETE /documents/{doc_id}`; uniform error body), `container.py` (lazy singletons: no AWS or Chroma access until first use).
- Config additions: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_UPLOAD_SIZE_MB`, `SUPPORTED_FILE_TYPES`, `EMBEDDING_BATCH_SIZE`. Collection-name validation added.
- Dependencies added: `chromadb`, `pymupdf`, `beautifulsoup4`, `fastapi`, `uvicorn`, `python-multipart` (runtime), `httpx` (dev, for TestClient). No LangChain.
- Tests: **185 passed** (`python -m pytest`, about 6s, offline; embeddings use a deterministic fake and mocked boto3).
- Manual check of the real server (no AWS): `/health` returned ok; `/ingest` with no credentials returned HTTP 502 `embedding_failed` with the credentials message and stored nothing; an unsupported `.docx` returned 400; a missing field returned 422.

### M1 design decisions (need approval)
1. **Extra stored metadata:** `content_hash`, `index_fingerprint`, `total_chunks` and `ingested_at` are stored in Chroma beside the contract fields, and are not part of `ChunkMetadata`. They are needed for re-ingestion. Adding them to the schema would be a contract change.
2. **Not-available values:** `""` and `-1` (page) are used instead of nullable fields, as the contract specifies.
3. **Document identity:** `doc_id` is a hash of the normalized lower-cased filename, so the same filename replaces the earlier document, and `a.md` from two folders would collide.
4. **No version extraction:** `version` is only the optional value passed to `/ingest` (it applies to every file in the request).
5. **Chunk text includes the `Document:`/`Section:` prefix** (embedded and stored). Citation excerpts are still substrings of `Chunk.text`. `CHUNK_SIZE` counts the body only.
6. **IngestResponse has no per-file status** (indexed/updated/unchanged), so an unchanged re-upload shows `chunks_indexed: 0`. *Proposal:* add an optional per-document `status` (or a `results` list). This is not done because it would change the contract.
7. **`VectorStore.query`** from contract §3 is left for M2.
8. **HTTP mapping of `/ingest`:** 200 if at least one file succeeded (failures listed). Otherwise 400 (all file problems), 502 (embedding failure) or 500. `DELETE` of an unknown id returns 200 with `deleted_chunks: 0`.
9. **Title rule:** PDF metadata title, else the sole level-1 heading, else the filename stem. Multiple H1s are deliberately not trusted as the document title.

### M1 known limitations
No OCR (scanned PDFs are rejected with a clear message); no PDF table reconstruction; PDF heading detection is a font heuristic; one Titan request per chunk (slow and throttle-prone on big corpora); ingests are serialized by a lock; the `.docx` and other formats are unsupported.

## M0 delivered (2026-11-14)
- Scaffold: `app/` packages, `scripts/`, `tests/`, `data/{raw,processed}`, `reports/`, `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.env.example`, `.gitignore` (appended `.venv*/`, `*.egg-info/`), `README.md`.
- `app/core/config.py`: pydantic-settings `Settings`, `load_settings()`, cached `get_settings()`, `ConfigurationError`. Titan V2 dimension is validated against 256/512/1024.
- `app/schemas/models.py`: all contract models. `app/core/schemas.py` re-exports them (see the schema-location note below).
- `app/core/bedrock.py`: `create_session`, `create_bedrock_runtime_client`, `BedrockRuntime` (`invoke_model`, `converse`), `translate_error` plus the `BedrockError` hierarchy.
- `scripts/test_bedrock.py`: opt-in live Titan and Nova checks.
- Tests: 53 passed in the local venv (config, schemas, mocked Bedrock client). No AWS or network used.
- Not done (out of M0 scope): embedding/generation logic, Chroma, ingestion, retrieval, API, frontend, evaluation.

## Done
- [x] Read the HCLTech problem statement and extracted FR-01 to FR-08.
- [x] Created `CLAUDE.md`, `docs/architecture.md`, `docs/api_contracts.md`, `docs/implementation_plan.md`, `docs/aws_prerequisites.md`, and this file.

## Milestone status
| Milestone | Owner | Status |
|---|---|---|
| M0 Foundations / contract freeze | M1 | done; live AWS check passes (ap-south-1) |
| M1 Ingestion & indexing | M2 + M1 | done; live Titan ingestion verified |
| M2 Retrieval | M3 | done; live-verified; thresholds still heuristic |
| M3 Generation | M4 | done; live Nova generation verified |
| M4 Integration & UI | M1 / M4 | done; Streamlit pages tested offline and against the live backend |
| M5 Evaluation & tuning | M4 | done; live evaluation run and analysed (report + before/after) |
| M6 Submission polish | all | README/docs updated; live verification done; chunk-size decision open |

## Decisions made
- Stack and flows as listed in `CLAUDE.md` (finalized by the user).
- Contracts in `api_contracts.md` are drafts, pending approval.

## Open decisions / assumptions to confirm
1. The embeddings and vectorstore wrappers are **assigned to Member 1** (assumption, since the brief's four areas do not name them).
2. Embedding dimension 1024.
3. Reranker approach (BM25 + vector RRF proposed) and conflict detection approach (metadata + heuristics proposed).
4. Source of the document corpus, and the version-extraction convention.
5. AWS account, region and credential mechanism (`docs/aws_prerequisites.md`).
6. How much of the pipeline uses LangChain abstractions vs. plain boto3 and chromadb (proposal: LangChain for splitters and prompt templates only where they do not obscure the contracts).

## Contract change log
- 2026-11-14 (M2, additive, needs approval): `ChromaStore.get_embeddings(ids)` was added and `get_all_chunks` gained an optional `where` argument (the contract lists `get_all_chunks()` with no arguments). A new HTTP route `POST /retrieve` was added. No schema field changed. See M2 decision 1.
- 2026-11-14 (needs approval): the contract names `app/core/schemas.py`, while the M0 brief specified an `app/schemas/` package. Models live in `app/schemas/`, and `app/core/schemas.py` re-exports them so both import paths work. No field changes. Added validation: `page` is `-1` or >= 1, `score` is 0..1, `vector_score` is -1..1, `rank` >= 1, non-empty IDs/text, and `QueryRequest.question` must not be whitespace-only. `DocumentInfo` fields come from the inline comment in contract §3.

## Next steps
1. Run the live smoke test (`python scripts/test_bedrock.py --all`) and record the result here, including the region and whether Nova needs an inference-profile ID.
2. Start the API, ingest a small real document set through `/ingest`, and check `/health` and `/documents`. Record the result and any throttling seen.
3. Approve or amend the contract-change note and the M1 decisions above.
4. M3 (generation, Member 4): `AnswerGenerator.generate(RetrievalResult) -> Answer` using Nova Lite via `BedrockRuntime.converse`. It must skip the LLM when `evidence_strength == "none"` (status `not_found`), present all sides when `conflicts.has_conflict`, and validate every citation against the returned `chunk_id`s. Then `POST /query` returning `QueryResponse`, which resolves M2 decision 1. Excerpts must be substrings of `Chunk.text`.
5. Before M5, calibrate `MIN_RELEVANCE` and the evidence margins on real Titan scores from the labelled set.

## Session notes
- 2026-11-14: The workspace initially contained only `docs/EnterpriseDocQ&Aassist_AIEngineer_Usecase2.pdf`. No files were overwritten. The PDF was read with `pdftotext`, because `pypdf` and the Read tool's PDF renderer were unavailable on this machine.
