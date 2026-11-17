"""A small local vector index: normalized float32 matrix plus the chunk documents.

Used for eval and for running the demo without Pinecone. `pinecone_store.py` serves production.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import INDEX_DIR


@dataclass
class Hit:
    doc: Document
    score: float


class LocalIndex:
    def __init__(self, docs: list[Document], matrix: np.ndarray, embedder: Embeddings | None = None):
        self.docs = docs
        self.matrix = matrix
        self.embedder = embedder

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        m = m.astype(np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)

    @classmethod
    def build(cls, docs, embedder: Embeddings, batch: int = 512, log=print) -> "LocalIndex":
        vecs = []
        for i in range(0, len(docs), batch):
            vecs.extend(embedder.embed_documents([d.page_content for d in docs[i : i + batch]]))
            log(f"  embedded {min(i + batch, len(docs))}/{len(docs)}")
        return cls(docs, cls._normalize(np.array(vecs)), embedder)

    @staticmethod
    def fingerprint(docs, embedder_name: str) -> str:
        """SHA-256 of the embedder name and every chunk text, in order."""
        h = hashlib.sha256()
        h.update(embedder_name.encode())
        for d in docs:
            h.update(b"\0")
            h.update(d.page_content.encode())
        return h.hexdigest()

    @classmethod
    def build_cached(cls, key: str, docs, embedder: Embeddings, log=print) -> "LocalIndex":
        path = INDEX_DIR / key
        name = getattr(embedder, "name", type(embedder).__name__)
        digest = cls.fingerprint(docs, name)
        meta = path / "meta.json"
        if meta.exists() and (path / "vectors.npy").exists():
            try:
                stored = json.loads(meta.read_text()).get("sha256")
            except (OSError, ValueError):
                stored = None
            if stored == digest:
                index = cls.load(path)
                index.embedder = embedder
                return index
        index = cls.build(docs, embedder, log=log)
        index.save(path)
        meta.write_text(json.dumps({"sha256": digest, "embedder": name, "chunks": len(docs)}) + "\n")
        return index

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.matrix)
        with open(path / "docs.jsonl", "w") as f:
            for d in self.docs:
                f.write(json.dumps({"text": d.page_content, "meta": d.metadata}) + "\n")

    @classmethod
    def load(cls, path: Path, embedder: Embeddings | None = None) -> "LocalIndex":
        matrix = np.load(path / "vectors.npy")
        docs = []
        with open(path / "docs.jsonl") as f:
            for line in f:
                row = json.loads(line)
                docs.append(Document(page_content=row["text"], metadata=row["meta"]))
        return cls(docs, matrix, embedder)

    def search_vector(self, vec: list[float], k: int = 5) -> list[Hit]:
        q = self._normalize(np.array(vec))
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self.docs[i], float(scores[i])) for i in top]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        assert self.embedder is not None, "index has no embedder"
        return self.search_vector(self.embedder.embed_query(query), k)
