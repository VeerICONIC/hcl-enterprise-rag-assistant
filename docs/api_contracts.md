# Interface Contracts

Single source of truth for the interfaces between members. Types live in `app/core/schemas.py` (pydantic v2). Change only with user approval, and log the change in `docs/PROGRESS.md`. Python signatures below are normative and the bodies are illustrative.

## 1. Shared schemas (`app/core/schemas.py`, owner: M1)

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    chunk_id: str            # deterministic: f"{doc_id}:v{version}:{ordinal:05d}"
    doc_id: str              # stable hash of normalized source path (or content)
    filename: str
    title: str               # document title ("" if unknown)
    section: str             # heading path, e.g. "Billing > Refunds"; "" if none
    page: int                # 1-based; -1 if not applicable (HTML/MD/TXT)
    version: str             # "" if unknown; else e.g. "2.1" or "2024-03-01"
    source: str              # path or URL of origin
    doc_type: Literal["pdf", "html", "md", "txt"]
    ordinal: int             # chunk order within document

class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float             # final score, higher = better, normalized 0..1
    vector_score: float      # cosine similarity
    rank: int                # 1-based after reranking

class ConflictGroup(BaseModel):
    topic: str
    chunk_ids: list[str]
    kind: Literal["contradiction", "version_difference", "date_difference"]
    description: str

class ConflictReport(BaseModel):
    has_conflict: bool
    groups: list[ConflictGroup] = []

class RetrievalResult(BaseModel):
    question: str
    chunks: list[RetrievedChunk]           # ordered best-first
    conflicts: ConflictReport
    evidence_strength: Literal["strong", "weak", "none"]
    top_score: float
    candidates_considered: int

class Citation(BaseModel):
    chunk_id: str
    filename: str
    title: str
    section: str
    page: int
    version: str
    excerpt: str             # verbatim substring of the chunk
    source: str              # link or identifier

class Answer(BaseModel):
    status: Literal["answered", "conflict", "needs_clarification", "not_found"]
    text: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    conflicts: ConflictReport
    clarifying_question: Optional[str] = None
    warnings: list[str] = []  # e.g. "removed 1 invalid citation"
```

## 2. Embeddings (`app/embeddings`, owner: M1)

```python
class Embedder(Protocol):
    dimension: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

- The implementation is `TitanEmbedder` (model `amazon.titan-embed-text-v2:0`, `normalize=True`, dimension from config). The same class serves both methods, with no model difference between docs and queries.
- Output order matches input order. It raises `EmbeddingError` after retries.

## 3. Vector store (`app/vectorstore`, owner: M1)

```python
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def delete_document(self, doc_id: str) -> int: ...
    def query(self, embedding: list[float], top_k: int,
              where: Optional[dict] = None) -> list[tuple[Chunk, float]]: ...  # (chunk, cosine_similarity)
    def list_documents(self) -> list[DocumentInfo]: ...
    def count(self) -> int: ...
    def get_chunks(self, doc_id: str) -> list[Chunk]: ...   # for BM25/conflict use by M3
    def get_all_chunks(self) -> list[Chunk]: ...            # for building a lexical index
```

`DocumentInfo = {doc_id, filename, title, version, doc_type, chunk_count}`.

## 4. Ingestion (`app/ingestion`, owner: M2)

```python
def ingest_file(path: Path, *, version: Optional[str] = None) -> list[Chunk]: ...
def ingest_paths(paths: list[Path]) -> list[Chunk]: ...
```

Guarantees: supports `.pdf/.html/.htm/.md/.markdown/.txt`. Every chunk has fully populated `ChunkMetadata` (use the sentinels `""` and `-1`, never `None`). `chunk_id` is deterministic. Chunk text is non-empty and bounded (<= Titan's 8192-token limit; target is far smaller). An unsupported or corrupt file raises `IngestionError(filename, reason)`. Ingestion does NOT call AWS. Embedding and storing are done by the API layer.

## 5. Retrieval (`app/retrieval`, owner: M3)

```python
class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, settings: RetrievalSettings): ...
    def retrieve(self, question: str, *, top_k: int = 5,
                 where: Optional[dict] = None) -> RetrievalResult: ...
```

Guarantees: `chunks` ordered best-first with `rank` set. `evidence_strength` is `none` when nothing clears the minimum-relevance threshold (the threshold is configurable and calibrated on the evaluation set). `conflicts` is always populated (`has_conflict=False` when none). Retrieval never calls an LLM unless explicitly enabled by the settings (default off). It does not import `app.generation`.

## 6. Generation (`app/generation`, owner: M4)

```python
class LLMClient(Protocol):
    def converse(self, system: str, messages: list[dict], *, max_tokens: int, temperature: float) -> str: ...

class AnswerGenerator:
    def __init__(self, llm: LLMClient, settings: GenerationSettings): ...
    def generate(self, retrieval: RetrievalResult) -> Answer: ...
```

Guarantees: deterministic fallback with no LLM call when `evidence_strength == "none"` (status `not_found`). Prompts contain only `retrieval.chunks` and use strict grounding rules. The model is asked for JSON `{status, answer, claims:[{text, evidence:[chunk_id]}], clarifying_question}`. After the call, `validate_citations` removes any chunk_id not in the retrieved set and checks that each excerpt is a substring of its chunk. When `conflicts.has_conflict`, the answer presents all sides with their citations (status `conflict`). The module has no dependency on `app.retrieval` internals, only on the schemas. The concrete `BedrockNovaClient` uses `bedrock-runtime.converse` with model `amazon.nova-lite-v1:0` (configurable).

## 7. HTTP API (`app/api`, owner: M1, consumed by the Streamlit frontend, M4)

| Method & path | Request | Response |
|---|---|---|
| `GET /health` | | `{status, chroma_count, embed_model, llm_model}` |
| `POST /ingest` | multipart `files[]` (+ optional `version`) | `IngestResponse` |
| `GET /documents` | | `list[DocumentInfo]` |
| `DELETE /documents/{doc_id}` | | `{deleted_chunks: int}` |
| `POST /query` | `QueryRequest` | `QueryResponse` |
| `POST /evaluate` (optional) | `{dataset_path}` | evaluation summary |

```python
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = 5
    doc_ids: Optional[list[str]] = None
    versions: Optional[list[str]] = None
    debug: bool = False

class QueryResponse(BaseModel):
    answer: Answer
    retrieved: Optional[list[RetrievedChunk]] = None  # present only if debug=True
    latency_ms: dict[str, int]                        # embed, retrieve, generate, total

class IngestResponse(BaseModel):
    documents: list[DocumentInfo]
    chunks_indexed: int
    failures: list[dict]                              # {filename, reason}
```

Errors use a uniform body `{"error": {"code": str, "message": str}}`, with HTTP codes 400 (bad input), 422 (validation), 502 (Bedrock failure), 500 (other).

## 8. Evaluation (`app/evaluation`, owner: M4)

Dataset (JSONL), one object per line:

```json
{"id": "q001", "question": "...", "category": "direct|paraphrase|conflict|ambiguous|unanswerable",
 "relevant_chunk_ids": ["..."], "expected_status": "answered|conflict|needs_clarification|not_found",
 "reference_answer": "...", "notes": "construction assumptions"}
```

Retrieval metrics: Recall@k, MRR, nDCG@k, hit rate. Answer metrics: groundedness (claim support check), citation precision/recall, fallback accuracy (`expected_status` match), conflict-surfacing rate, and an optional LLM-as-judge for usefulness (judge prompt and its limits documented). The evaluation calls the same `Retriever` and `AnswerGenerator` objects as the API, and emits `reports/eval_report.md` plus raw JSON.

## 9. Configuration contract (`app/core/config.py`, owner: M1)

Env vars (names only; values in the git-ignored `.env`, see `.env.example`):
`AWS_REGION`, `AWS_PROFILE` (optional), `BEDROCK_EMBED_MODEL_ID`, `BEDROCK_EMBED_DIM`, `BEDROCK_LLM_MODEL_ID`, `CHROMA_PATH`, `CHROMA_COLLECTION`, `RETRIEVAL_TOP_N`, `MIN_RELEVANCE`, `API_BASE_URL` (frontend).
No AWS key variable is read by application code. boto3's default chain resolves credentials.
