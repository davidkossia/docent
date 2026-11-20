from langchain_core.documents import Document

from docent.index import Hit
from docent.pinecone_store import PineconeRetriever, namespace_for, to_hit, upsert_metadata

META = {"chunk_id": "section-800:7", "book": "py", "book_title": "Intro to Python", "section_id": "py-1.5",
        "section_title": "1.5 Number basics", "section_ids": ["py-1.5", "py-1.6"], "page_start": 29, "page_end": 30}


class FakeStore:
    """Mimics PineconeVectorStore: numbers come back as floats and the text key is not in metadata."""

    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, query, k=4):
        self.calls.append((query, k))
        meta = {**META, "page_start": 29.0, "page_end": 30.0}
        return [(Document(page_content="Multiplication goes first.", metadata=meta), 0.83)]


def test_retriever_returns_hits_with_local_metadata_keys_and_types():
    store = FakeStore()
    hits = PineconeRetriever(store).search("what goes first?", 3)
    assert store.calls == [("what goes first?", 3)]
    h = hits[0]
    assert isinstance(h, Hit) and h.score == 0.83
    assert h.doc.page_content == "Multiplication goes first."
    assert h.doc.metadata == META
    assert type(h.doc.metadata["page_start"]) is int


def test_to_hit_leaves_other_metadata_alone():
    h = to_hit(Document(page_content="t", metadata={"chunk_id": "a:1", "page_start": 3.0}), 1)
    assert h.doc.metadata == {"chunk_id": "a:1", "page_start": 3}


def test_upsert_uses_chunk_id_and_keeps_every_metadata_key():
    docs = [Document(page_content="one", metadata=dict(META)), Document(page_content="two", metadata={**META, "chunk_id": "section-800:8"})]
    ids, texts, metas = upsert_metadata(docs)
    assert ids == ["section-800:7", "section-800:8"]
    assert texts == ["one", "two"]
    assert metas[0] == META and metas[0] is not docs[0].metadata


def test_namespace():
    assert namespace_for("section-800", "local-bge-small") == "section-800__local-bge-small"
