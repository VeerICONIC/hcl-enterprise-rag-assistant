"""Lazily-built, process-wide service container.

Nothing here touches AWS or ChromaDB until first use, so importing the app, and
endpoints that do not need them, never require credentials. Each component is built
once and reused by every request.
"""

from __future__ import annotations

import threading
from typing import Optional

from app.core.config import Settings, get_settings
from app.embeddings.titan import TitanEmbedder
from app.ingestion.ingestion_service import IngestionService
from app.api.query_service import QueryService
from app.generation import AnswerGenerator, BedrockNovaClient, GenerationSettings
from app.retrieval import Retriever, RetrievalSettings
from app.vectorstore.chroma_store import ChromaStore


class Container:
    def __init__(self, settings: Optional[Settings] = None, *, store=None, embedder=None, service=None, retriever=None, generator=None, query_service=None):
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._service = service
        self._retriever = retriever
        self._generator = generator
        self._query_service = query_service
        self._lock = threading.Lock()

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def store(self):
        with self._lock:
            if self._store is None:
                self._store = ChromaStore(self.settings)
            return self._store

    @property
    def embedder(self):
        with self._lock:
            if self._embedder is None:
                self._embedder = TitanEmbedder(settings=self.settings)
            return self._embedder

    @property
    def ingestion(self) -> IngestionService:
        store, embedder = self.store, self.embedder  # resolved outside the lock below
        with self._lock:
            if self._service is None:
                self._service = IngestionService(embedder, store, self.settings)
            return self._service

    @property
    def retriever(self) -> Retriever:
        store, embedder = self.store, self.embedder
        with self._lock:
            if self._retriever is None:
                self._retriever = Retriever(embedder, store, RetrievalSettings.from_settings(self.settings))
            return self._retriever

    @property
    def generator(self) -> AnswerGenerator:
        with self._lock:
            if self._generator is None:
                self._generator = AnswerGenerator(BedrockNovaClient(settings=self.settings),
                                                  GenerationSettings.from_settings(self.settings))
            return self._generator

    @property
    def query_service(self) -> QueryService:
        retriever, generator = self.retriever, self.generator
        with self._lock:
            if self._query_service is None:
                self._query_service = QueryService(retriever, generator)
            return self._query_service
