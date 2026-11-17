"""Embedders behind LangChain's `Embeddings` interface.

Providers:
  local-bge-small   fastembed, BAAI/bge-small-en-v1.5 (384 dims), no key
  local-bge-base    fastembed, BAAI/bge-base-en-v1.5 (768 dims), no key
  openai-3-small    OpenAI text-embedding-3-small (1536 dims), needs OPENAI_API_KEY
  voyage            Voyage AI voyage-3.5-lite, needs VOYAGE_API_KEY and the `voyage` extra
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import DATA_DIR, env

LOCAL_MODELS = {
    "local-bge-small": "BAAI/bge-small-en-v1.5",
    "local-bge-base": "BAAI/bge-base-en-v1.5",
}
NAMES = [*LOCAL_MODELS, "openai-3-small", "voyage"]


class LocalEmbeddings(Embeddings):
    def __init__(self, name: str):
        from fastembed import TextEmbedding

        self.name = name
        self._model = TextEmbedding(LOCAL_MODELS[name], cache_dir=str(DATA_DIR / "models"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for v in self._model.embed(texts, batch_size=64):
            out.append(v.tolist())
            if len(out) % 256 == 0 or len(out) == len(texts):
                print(f"    {self.name}: {len(out)}/{len(texts)}", flush=True)
        return out

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class VoyageEmbeddings(Embeddings):
    name = "voyage"
    model = "voyage-3.5-lite"

    def __init__(self):
        import voyageai

        key = env("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        self._client = voyageai.Client(api_key=key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            r = self._client.embed(texts[i : i + 128], model=self.model, input_type="document")
            out.extend(r.embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed([text], model=self.model, input_type="query").embeddings[0]


def get_embedder(name: str) -> Embeddings:
    if name in LOCAL_MODELS:
        return LocalEmbeddings(name)
    if name == "openai-3-small":
        from langchain_openai import OpenAIEmbeddings

        key = env("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        e = OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)
        e.name = name  # type: ignore[attr-defined]
        return e
    if name == "voyage":
        return VoyageEmbeddings()
    raise ValueError(f"unknown embedder {name!r}; choose from {NAMES}")
