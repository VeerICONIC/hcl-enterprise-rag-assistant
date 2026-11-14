"""FastAPI application. Run with: ``uvicorn app.api.main:app``.

M1 routes (api_contracts.md §7): GET /health, POST /ingest, GET /documents,
DELETE /documents/{doc_id}. Errors use ``{"error": {"code", "message"}}``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.container import Container
from app.core.bedrock import BedrockError
from app.core.config import ConfigurationError
from app.generation import GenerationError
from app.ingestion.ingestion_service import FileFailure
from app.retrieval import InvalidQueryError, build_where
from app.schemas import DocumentInfo, IngestResponse, QueryRequest, QueryResponse, RetrievalResult
from app.vectorstore.chroma_store import VectorStoreError

logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def get_container(request: Request) -> Container:
    return request.app.state.container


def create_app(container: Optional[Container] = None) -> FastAPI:
    app = FastAPI(title="Enterprise Documentation Q&A Assistant", version="0.1.0")
    app.state.container = container or Container()

    # -- error handling: uniform body, never a stack trace ------------------
    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        parts = [f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()]
        return _error(422, "validation_error", "; ".join(parts) or "invalid request")

    @app.exception_handler(InvalidQueryError)
    async def _invalid_query(_: Request, exc: InvalidQueryError):
        return _error(400, "invalid_query", str(exc))

    @app.exception_handler(GenerationError)
    async def _generation(_: Request, exc: GenerationError):
        logger.error("generation error: %s", exc)
        return _error(502, "generation_failed", str(exc))

    @app.exception_handler(ConfigurationError)
    async def _config(_: Request, exc: ConfigurationError):
        logger.error("configuration error: %s", exc)
        return _error(500, "configuration_error", str(exc))

    @app.exception_handler(BedrockError)
    async def _bedrock(_: Request, exc: BedrockError):
        logger.error("bedrock error: %s", exc)
        return _error(502, "bedrock_error", str(exc))

    @app.exception_handler(VectorStoreError)
    async def _store(_: Request, exc: VectorStoreError):
        return _error(500, "vector_store_error", str(exc))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return _error(500, "internal_error", "internal server error")

    # -- routes --------------------------------------------------------------
    @app.get("/health")
    def health(c: Container = Depends(get_container)):
        s = c.settings
        try:
            chunks = c.store.count()
        except VectorStoreError as exc:
            return _error(503, "vector_store_unavailable", str(exc))
        return {"status": "ok", "chroma_count": chunks, "embed_model": s.bedrock_embed_model_id,
                "llm_model": s.bedrock_llm_model_id}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(
        files: list[UploadFile] = File(..., description="PDF, HTML, Markdown or TXT files"),
        version: Optional[str] = Form(None, description="Optional version label applied to these files"),
        c: Container = Depends(get_container),
    ):
        limit = c.settings.max_upload_bytes
        to_ingest: list[tuple[str, bytes]] = []
        oversize: list[str] = []
        for f in files:
            data = await f.read(limit + 1)  # never buffer more than limit+1 bytes
            if len(data) > limit:
                oversize.append(f.filename or "")
            else:
                to_ingest.append((f.filename or "", data))

        batch = await run_in_threadpool(c.ingestion.ingest_files, to_ingest, version=version)
        for name in oversize:
            batch.failures.append(FileFailure(
                name, "file_too_large",
                f"file exceeds the {c.settings.max_upload_size_mb} MB upload limit", True))

        if not batch.outcomes and batch.failures:
            codes = {f.code for f in batch.failures}
            status = 502 if "embedding_failed" in codes else (400 if all(f.client_error for f in batch.failures) else 500)
            message = "; ".join(f"{f.filename or '(unnamed)'}: {f.reason}" for f in batch.failures)
            return _error(status, batch.failures[0].code, message)
        return batch.to_response()

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, c: Container = Depends(get_container)):
        """Grounded answer: retrieval + Nova Lite generation with validated citations."""
        return c.query_service.query(req)

    @app.post("/retrieve", response_model=RetrievalResult)
    def retrieve(req: QueryRequest, c: Container = Depends(get_container)):
        """Retrieved evidence only (no generation); useful for debugging retrieval. ``POST /query`` adds generation."""
        return c.retriever.retrieve(req.question, top_k=req.top_k,
                                    where=build_where(req.doc_ids, req.versions))

    @app.get("/evaluation/latest")
    def latest_evaluation(c: Container = Depends(get_container)):
        """Most recent evaluation results written by ``scripts/run_evaluation.py`` (read-only)."""
        path = Path(c.settings.reports_dir) / "eval_results.json"
        if not path.is_file():
            return _error(404, "no_evaluation", "No evaluation results yet. Run: python scripts/run_evaluation.py")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _error(500, "evaluation_unreadable", "The evaluation results file could not be read.")

    @app.get("/evaluation/report")
    def evaluation_report(c: Container = Depends(get_container)):
        """The Markdown evaluation report (read-only)."""
        path = Path(c.settings.reports_dir) / "evaluation_report.md"
        if not path.is_file():
            return _error(404, "no_evaluation", "No evaluation report yet. Run: python scripts/run_evaluation.py")
        try:
            return {"markdown": path.read_text(encoding="utf-8")}
        except OSError:
            return _error(500, "evaluation_unreadable", "The evaluation report could not be read.")

    @app.get("/documents", response_model=list[DocumentInfo])
    def documents(c: Container = Depends(get_container)):
        return c.ingestion.list_documents()

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str, c: Container = Depends(get_container)):
        return {"deleted_chunks": c.ingestion.delete_document(doc_id)}

    return app


app = create_app()
