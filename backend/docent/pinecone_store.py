"""Pinecone index for production retrieval: `PineconeRetriever` and the `index` command's logic."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import env
from .index import Hit

INDEX_NAME = "docent"
CLOUD, REGION = "aws", "us-east-1"
DIMENSIONS = {"local-bge-small": 384, "local-bge-base": 768, "openai-3-small": 1536, "voyage": 1024}
INT_KEYS = ("page_start", "page_end")  # Pinecone returns numbers as floats


def namespace_for(strategy: str, embedder: str) -> str:
    return f"{strategy}__{embedder}"


def _client():
    from pinecone import Pinecone

    key = env("PINECONE_API_KEY")
    if not key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=key)


def _store(embedder: Embeddings, namespace: str):
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(
        index=_client().Index(INDEX_NAME), embedding=embedder, namespace=namespace, text_key="text"
    )


def to_hit(doc: Document, score: float) -> Hit:
    """Map a stored vector's document back to a `Hit` with the metadata types chunks have locally."""
    meta = dict(doc.metadata)
    for key in INT_KEYS:
        if key in meta:
            meta[key] = int(meta[key])
    return Hit(Document(page_content=doc.page_content, metadata=meta), float(score))


class PineconeRetriever:
    """Same `search(question, k)` as `LocalRetriever`, backed by a Pinecone namespace."""

    def __init__(self, store):
        self.store = store

    @classmethod
    def connect(cls, strategy: str, embedder_name: str, embedder: Embeddings) -> "PineconeRetriever":
        return cls(_store(embedder, namespace_for(strategy, embedder_name)))

    def search(self, question: str, k: int = 5) -> list[Hit]:
        return [to_hit(d, s) for d, s in self.store.similarity_search_with_score(question, k=k)]


def upsert_metadata(docs: list[Document]) -> tuple[list[str], list[str], list[dict]]:
    """ids, texts and metadata for upsert: `chunk_id` is the vector id; metadata holds every chunk key."""
    return (
        [d.metadata["chunk_id"] for d in docs],
        [d.page_content for d in docs],
        [dict(d.metadata) for d in docs],
    )


def index_docs(docs: list[Document], embedder: Embeddings, strategy: str, embedder_name: str,
               create: bool = False, log=print) -> None:
    """Embed `docs` and upsert them. Creates the index only when `create` is true."""
    namespace = namespace_for(strategy, embedder_name)
    if create:
        from pinecone import ServerlessSpec

        _client().create_index(
            name=INDEX_NAME, dimension=DIMENSIONS[embedder_name], metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        log(f"created index {INDEX_NAME} ({CLOUD} {REGION}, cosine)")
    ids, texts, metas = upsert_metadata(docs)
    _store(embedder, namespace).add_texts(texts, metadatas=metas, ids=ids)
    log(f"upserted {len(ids)} chunks into {INDEX_NAME}/{namespace}")
