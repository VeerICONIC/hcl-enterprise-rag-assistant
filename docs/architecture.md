# Architecture

## 1. Requirements (from the HCLTech brief)

| ID | Capability | Requirement (summary) | Module(s) |
|---|---|---|---|
| FR-01 | Multi-format ingestion | Ingest PDF, HTML, Markdown and TXT; preserve title, section and source metadata | `ingestion` |
| FR-02 | Content preparation | Extract, clean and segment into passages without losing headings or context | `ingestion` |
| FR-03 | NL retrieval | Retrieve and rank passages, handling paraphrases and terminology variation | `embeddings`, `vectorstore`, `retrieval` |
| FR-04 | Grounded answers | Concise answer using only retrieved evidence | `generation` |
| FR-05 | Citations | Source doc, section/page, excerpt, and link/identifier per claim | `generation` (validation), metadata from `ingestion` |
| FR-06 | Conflict handling | Detect contradictory or version-sensitive passages and present the conflict | `retrieval` (detect), `generation` (present) |
| FR-07 | Confidence & fallback | Flag weak evidence, ask for context, or return a clear not-found | `retrieval` (signals), `generation` (decision) |
| FR-08 | Evaluation | Measure retrieval and answer quality on a labelled set (answerable, ambiguous, unanswerable) | `evaluation` |

Submission deliverables: a working prototype (web UI or API), an evaluation report, at least 5 evidence-backed examples (direct, paraphrase, conflict, unsupported), and a README.

Evaluation dimensions: retrieval quality, groundedness, usefulness, conflict handling, fallback behaviour. The synthetic labelled set and its assumptions must be documented.

## 2. Components

```
                      ┌────────────────────────┐
                      │ Streamlit frontend (M4)│
                      └───────────┬────────────┘
                                  │ HTTP/JSON
                      ┌───────────▼────────────┐
                      │ FastAPI (M1)           │  only layer that wires modules
                      └─┬────────┬──────────┬──┘
          ingest        │        │ query    │
   ┌────────────────────▼─┐   ┌──▼──────────▼──┐   ┌───────────────────┐
   │ ingestion (M2)       │   │ retrieval (M3) │   │ generation (M4)   │
   │ parse/clean/chunk    │   │ search/rerank/ │   │ prompt/Nova Lite/ │
   └──────────┬───────────┘   │ conflicts      │   │ cite-validate/    │
              │ Chunk[]       └───────┬────────┘   │ fallback          │
   ┌──────────▼───────────┐           │            └─────────┬─────────┘
   │ embeddings (M1)      │◄──────────┘ (query embed)        │
   │ Titan V2 via Bedrock │                                  │
   └──────────┬───────────┘                                  │
   ┌──────────▼───────────┐                       ┌──────────▼─────────┐
   │ vectorstore (M1)     │                       │ Bedrock Converse   │
   │ ChromaDB persistent  │                       │ Nova Lite          │
   └──────────────────────┘                       └────────────────────┘
```

Independence rule: `ingestion`, `retrieval` and `generation` import only `app.core.schemas`. Retrieval receives an `Embedder` and `VectorStore` by injection (protocols), and generation receives an `LLMClient`. `app/api` constructs and injects the concrete implementations.

## 3. Data flows

### Ingestion
1. `POST /ingest` receives files (or a directory path in `data/raw`).
2. **Parse** by extension into `ParsedDocument` (pages/sections with text and headings). Supported: PDF, HTML, MD, TXT.
3. **Preprocess**: normalize whitespace and unicode, strip boilerplate (headers/footers/nav), dehyphenate, and keep the heading hierarchy as a `section` path.
4. **Chunk**: heading-aware splitting with overlap. Each chunk keeps its section path, page and a contextual prefix (title › section) to preserve context.
5. Compute `doc_id` (stable hash of source path or content), `version` (parsed from the doc or filename, otherwise a content hash or the supplied value), and a deterministic `chunk_id`.
6. **Embed** the chunk text with Titan V2 (bounded concurrency, retries).
7. **Upsert** into ChromaDB (`documents` = chunk text, `embeddings`, `metadatas` = the required metadata, `ids` = `chunk_id`). Re-ingesting the same document is idempotent (delete by `doc_id`, then add).

### Query
1. `POST /query` receives `{question, filters?, top_k?}`.
2. Embed the question with Titan V2 (same model and dimension as the documents).
3. **Retrieve** the top-N candidates from Chroma (cosine). Optional metadata filters (`doc_id`, `version`).
4. **Rerank** and trim to top-k (see the open decision on the approach). Compute evidence-strength signals.
5. **Conflict detection** across the retrieved set: same topic with differing values, or the same `doc_id` in different `version`s, or explicit supersedes/deprecated language. Emits a `ConflictReport`.
6. **Generate** with Nova Lite via Converse. The prompt contains only the numbered evidence and strict grounding rules, and asks for structured JSON output (answer, claims with evidence IDs).
7. **Validate citations**: every cited ID must exist in the retrieved set. Drop invalid ones, and downgrade or fall back if material claims lose all support.
8. **Fallback decision**: `answered` | `conflict` | `needs_clarification` | `not_found`, with a confidence label.
9. Return `QueryResponse` (answer, citations with excerpt and identifier, conflicts, status, confidence, debug trace).

## 4. Key decisions

| Decision | Rationale |
|---|---|
| Titan V2 for docs and queries | The same vector space is required. V2 is configurable in dimension and normalizes output. |
| Chroma instead of Bedrock KB | Full control over metadata, reranking and conflict logic, which the brief rewards. |
| Nova Lite via Converse | A cheap, fast model with a unified API. Converse keeps the client model-agnostic. |
| Structured JSON generation | Enables deterministic citation validation instead of parsing prose. |
| Heuristic + LLM-assisted conflict detection | See the open decisions. |
| Deterministic `chunk_id` | Idempotent re-ingest and stable evaluation labels. |
| Credentials from the boto3 default chain | Satisfies the no-hardcoding rule. |

## 5. Open design decisions (resolve in the milestone noted)

- **Reranker** (M3): lexical hybrid (BM25 + vector fusion), MMR, or a Bedrock rerank model. Default proposal: vector search + BM25 fusion (RRF), with no extra AWS model.
- **Conflict detector** (M3): rule/metadata-based (versions, dates) plus an optional cheap LLM pairwise check. Default proposal: metadata + numeric/negation heuristics first.
- **Chunk size** (M2): proposal of ~500 to 800 tokens with 10–15% overlap, tuned on the evaluation set.
- **Embedding dimension** (M1): proposal of 1024 for quality (512 if size or latency matters). Fixed once indexed.
- **LangChain scope**: use it for orchestration glue (text splitters, prompt templates, Bedrock/Chroma integrations where they do not hide the contract). The module interfaces stay plain pydantic and protocols.
- **Version extraction**: from document metadata, filename (`v2.1`), or release-note dates. A convention is needed from the dataset.
- **Document set**: no sample corpus yet. Needs to be supplied or synthesized (including deliberately conflicting docs).
