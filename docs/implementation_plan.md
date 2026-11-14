# Implementation Plan

Milestones are ordered by dependency. Members work in parallel after M0, and M0 locks the contracts. Items marked [mock] let a member proceed before another member's module exists.

## M0: Foundations (M1, all review)
- Repo scaffold: package layout, `pyproject.toml`/`requirements.txt`, `.gitignore` (`.env`, `data/chroma`), `.env.example`, pre-commit/ruff, pytest config.
- `app/core`: `config.py` (pydantic-settings), `schemas.py` (from `api_contracts.md`), `aws.py` (boto3 session via default chain), logging.
- Verify AWS access with a smoke script (Titan embed + Nova Converse call). See `aws_prerequisites.md`.
- Create or collect the sample corpus (PDF/HTML/MD/TXT) including intentionally conflicting and versioned docs. Everyone needs this.
- **Exit:** the contracts are approved by all four members, the smoke test passes, and the scaffold is merged.

## M1: Ingestion and indexing path (M2 + M1)
- M2: parsers (PDF with page numbers, HTML with headings, MD, TXT), cleaning, heading-aware chunker, deterministic IDs, version extraction. Unit tests per format.
- M1: `TitanEmbedder` (concurrency, retry/backoff), `ChromaStore` (cosine, scalar metadata, idempotent upsert), `POST /ingest`, `GET /documents`, `DELETE /documents/{id}`.
- **Exit:** the sample corpus is ingested end-to-end, the Chroma contents are verified against the metadata contract, and re-ingestion is idempotent.

## M2: Retrieval (M3)
*Status: implemented 2026-11-14 as BM25 + vector RRF, evidence heuristics, and metadata/heuristic conflict detection (see `docs/PROGRESS.md`). Thresholds still need calibration (M5).*
- Vector retrieval through `Embedder` and `VectorStore` [mock the store with in-memory fixtures until M1 lands].
- Reranking (decision recorded in `PROGRESS.md`), score normalization, evidence-strength thresholds.
- Conflict detection (version/date/value contradictions) with a labelled test fixture.
- **Exit:** `Retriever.retrieve` meets the contract, with unit tests for ranking, thresholds and conflicts.

## M3: Generation (M4)
*Status: implemented 2026-11-14 (offline-tested; the live Nova call is blocked by the AWS account hold).*
- `BedrockNovaClient` (Converse), prompt templates, structured JSON parsing with repair/retry.
- Citation validation, excerpt-substring check, fallback logic (`not_found`, `needs_clarification`), conflict presentation.
- Developed against hand-built `RetrievalResult` fixtures [mock] and a fake `LLMClient` for tests.
- **Exit:** `AnswerGenerator.generate` meets the contract, and adversarial tests (hallucinated citation IDs, empty evidence, conflicting evidence) pass.

## M4: Integration (M1 lead)
*Status: implemented 2026-11-14 (Streamlit pages plus `POST /query`).*
- `POST /query` wiring (embed → retrieve → generate), latency breakdown, uniform error handling, `/health`.
- Streamlit frontend (M4): upload, ask, answer with expandable citations/excerpts, a visible conflict panel, a fallback display, and a debug view.
- **Exit:** the five brief example questions (direct, paraphrase, conflict, unsupported, plus one more) work through the UI.

## M5: Evaluation and tuning (M4 lead, everyone contributes)
*Status: framework implemented 2026-11-14; only an offline dry run exists, and the live run is pending AWS access.*
- Labelled dataset: >= ~30 questions across direct, paraphrase, conflict, ambiguous and unanswerable categories, with construction assumptions documented. Relevance labels are by `chunk_id`.
- Metrics runner and report (retrieval and answer metrics, failure cases, limitations).
- Tune chunk size, top-k, thresholds and the reranker against the metrics, and record the results.
- **Exit:** `reports/eval_report.md` is generated reproducibly.

## M6: Submission polish (all)
*Status: README and docs done; live verification pending.*
- README (setup, supported formats, indexing/retrieval approach, assumptions, trade-offs), the evaluation report, the evidence-backed examples (>= 5), and a demo script.
- Clean-clone run-through, with no secrets in the repo.

## Cross-cutting
- Tests never call AWS by default (mock Bedrock). Live tests need an explicit marker.
- Every contract change gets logged in `docs/PROGRESS.md`.
- Suggested sync points: the end of M0 (contract freeze), the end of M2/M3 (integration check), and before M5 (the dataset must be final enough to label).
