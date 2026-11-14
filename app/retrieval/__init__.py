"""Retrieval (M2): vector + BM25 candidates, RRF fusion, evidence strength, conflict detection."""

from app.retrieval.models import InvalidQueryError, RetrievalSettings
from app.retrieval.retriever import Retriever, build_where

__all__ = ["InvalidQueryError", "RetrievalSettings", "Retriever", "build_where"]
