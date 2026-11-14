# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HCLTech Future Ready AI Engineer Hackathon, Use Case 2: **Enterprise Documentation Q&A Assistant**. It ingests PDF/HTML/Markdown/TXT documents and answers natural-language questions using *only* retrieved evidence, with citations, conflict handling, confidence/fallback, and a measured evaluation. The source brief is `docs/EnterpriseDocQ&Aassist_AIEngineer_Usecase2.pdf`. It defines FR-01 to FR-08, which are mapped to modules in `docs/architecture.md`.

Scope boundary from the brief: the assistant answers and cites. It never modifies source documents or enterprise systems.

**Status:** M0 to M6 implemented and offline-tested (431 tests). Live Bedrock works (Titan V2 + Nova Lite via `apac.amazon.nova-lite-v1:0`, `ap-south-1`) and a live evaluation exists; see `docs/PROGRESS.md` for the analysis and the open chunk-size decision. Read `docs/PROGRESS.md` first in every session and update it before ending a session.

## Documentation map

- `docs/architecture.md`: architecture, data flows, FR mapping, design decisions
- `docs/api_contracts.md`: **the interface contract between the four members. Treat it as the source of truth for types and signatures.**
- `docs/implementation_plan.md`: milestones and ownership
- `docs/aws_prerequisites.md`: AWS setup and configuration checklist
- `docs/PROGRESS.md`: living progress log and open decisions

## Finalized stack (do not change without asking the user)

Python, FastAPI, LangChain (orchestration), Amazon Bedrock Runtime, **Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`) for both document and query embeddings, **Amazon Nova Lite** (`amazon.nova-lite-v1:0`) through the Bedrock **Converse API** for generation, **ChromaDB** (persistent) as the vector store, Streamlit frontend, custom evaluation framework.

Flows:
- Ingestion: Documents → Parsing → Preprocessing → Chunking → Titan embeddings → ChromaDB
- Query: Question → Titan query embedding → ChromaDB retrieval → evidence → Nova Lite → grounded answer with citations

## Permanent rules

1. Titan embeds both documents and queries. Never mix embedding models or dimensions within a collection.
2. Use ChromaDB for retrieval. **Do not use Bedrock Knowledge Bases.**
3. Every chunk carries the metadata: `filename`, `page`, `section`, `doc_id`, `version`, `chunk_id` (plus `title`, `source`). Never drop or rename these fields. See `api_contracts.md`.
4. Generated answers must use only retrieved evidence. Every material claim needs a citation to a retrieved chunk. Citations are validated after generation, and any citation that does not map to a retrieved `chunk_id` is removed or the answer is rejected.
5. Conflicting or version-sensitive evidence is surfaced to the user, never silently resolved.
6. Insufficient or weak evidence returns an explicit fallback (`not_found` or `needs_clarification`) and never a guess.
7. **Never hardcode AWS credentials**, and never commit `.env`, keys or profiles. Use the boto3 default credential chain (SSO profile, env vars or IAM role). Only region, model IDs and profile *names* are configured, via `.env` and `app/core/config.py`.
8. Keep embedding, retrieval and generation modules independent. They communicate only through the pydantic models in `api_contracts.md`. Retrieval must not import generation (and vice versa), and only `app/api` wires them together.
9. Respect module ownership (below). Changing a shared contract needs the user's approval, and the change must be logged in `docs/PROGRESS.md`.

## Team ownership

| Member | Area | Package(s) |
|---|---|---|
| 1 | Backend & integration (the user) | `app/api`, `app/core`, `app/embeddings`, `app/vectorstore` |
| 2 | Ingestion & chunking | `app/ingestion` |
| 3 | Retrieval, reranking, conflict detection | `app/retrieval` |
| 4 | Generation, citations, frontend, evaluation | `app/generation`, `app/evaluation`, `frontend/` |

The embeddings/vectorstore assignment to Member 1 is an assumption and is pending confirmation (see `docs/PROGRESS.md`).

## Planned layout

```
app/
  api/          FastAPI app, routes, DI wiring
  core/         config (pydantic-settings), boto3 client factory, logging, shared schemas
  ingestion/    parsers, cleaning, chunking
  embeddings/   Titan V2 wrapper
  vectorstore/  Chroma wrapper
  retrieval/    search, rerank, conflict detection
  generation/   prompt, Nova Lite via Converse, citation validation, fallback
  evaluation/   labelled set, retrieval + answer metrics, report
frontend/       Streamlit app
data/           raw docs, chroma persistence (git-ignored)
tests/
```

## Commands

Windows, venv at `.venv` (Python 3.12). Setup details are in `README.md`.

- Install: `python -m pip install -r requirements-dev.txt`
- Tests (offline, no AWS): `python -m pytest`, or a single test: `python -m pytest tests/test_config.py::test_defaults`
- Live Bedrock smoke test (opt-in, billable): `python scripts/test_bedrock.py --all`
- Run the API: `uvicorn app.api.main:app --port 8000` (routes: `/health`, `/ingest`, `/documents`, `/documents/{doc_id}`, `/query`, `/retrieve`, `/evaluation/latest`)
- Run the UI: `streamlit run frontend/app.py --server.address localhost --server.port 8501`; or `powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1` / `start_ui.ps1` / `start_all.ps1` / `stop_all.ps1`
- Evaluation: `python scripts/run_evaluation.py` (offline dry run) or `--live` (billable)

Generation notes: `AnswerGenerator` never calls the LLM without evidence, builds citations only from retrieved chunks, and withholds answers with no valid citation. The Streamlit app talks only to FastAPI (never Bedrock or Chroma directly). Evaluation indexes the synthetic corpus into a temporary store and labels offline runs as non-representative. `.env` sets `AWS_PROFILE=hcl-rag` (profile name only).

Retrieval notes: `Retriever` (`app/retrieval/retriever.py`) fuses vector and BM25 results with RRF and returns the contract's `RetrievalResult`; it never calls an LLM. `POST /retrieve` is a temporary M2 route (the contract's `POST /query` needs M3's `Answer`). Evidence thresholds and conflict heuristics are provisional and uncalibrated. M3 must check `evidence_strength` before using `chunks`.

Ingestion notes: `IngestionService` (`app/ingestion/ingestion_service.py`) holds the pipeline and is independent of FastAPI. Chroma holds extra per-chunk bookkeeping (`content_hash`, `index_fingerprint`, `total_chunks`, `ingested_at`) beside the contract metadata. Embedding is done before the store is touched, so failed re-ingests keep the old version. Details are in `README.md` and `docs/PROGRESS.md`. Add them when those milestones land, and do not invent them. Schemas live in `app/schemas/`, and `app/core/schemas.py` re-exports them.

## Gotchas to remember

- Chroma metadata values must be scalars (str/int/float/bool). Do not store `None` or lists. Use `""` or `-1` sentinels and flatten.
- Create the Chroma collection with cosine distance (`hnsw:space = cosine`) and convert distance to similarity consistently.
- Titan V2 supports 256/512/1024 dimensions. The chosen dimension is fixed in config, and changing it requires re-indexing.
- Titan has no native batch call, so use bounded concurrency and retry with backoff on throttling.
- Nova Lite may require a cross-region inference profile ID (e.g. `us.amazon.nova-lite-v1:0`) depending on region. Make the model ID configurable.
- Unit tests must not call AWS. Mock the Bedrock clients, and gate any live tests behind an explicit marker.
