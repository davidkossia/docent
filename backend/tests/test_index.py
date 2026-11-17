import json

import numpy as np
from langchain_core.documents import Document

from docent import index as index_mod
from docent.index import LocalIndex


class FakeEmbedder:
    """Maps each text to a vector from its letter counts. Counts calls."""

    name = "fake"

    def __init__(self):
        self.doc_calls = 0

    def _vec(self, text: str) -> list[float]:
        return [text.lower().count(c) + 0.01 for c in "abcdef"]

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _docs(*texts):
    return [Document(page_content=t, metadata={"section_ids": [f"s{i}"]}) for i, t in enumerate(texts)]


def test_build_and_search_order():
    emb = FakeEmbedder()
    idx = LocalIndex.build(_docs("aaaa", "bbbb", "cccc"), emb, log=lambda *_: None)
    hits = idx.search("bbb", k=3)
    assert hits[0].doc.page_content == "bbbb"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_save_and_load_roundtrip(tmp_path):
    emb = FakeEmbedder()
    idx = LocalIndex.build(_docs("aaaa", "bbbb"), emb, log=lambda *_: None)
    idx.save(tmp_path / "i")
    loaded = LocalIndex.load(tmp_path / "i", emb)
    assert np.allclose(loaded.matrix, idx.matrix)
    assert [d.page_content for d in loaded.docs] == ["aaaa", "bbbb"]
    assert loaded.search("aaa", k=1)[0].doc.page_content == "aaaa"


def test_build_cached_reuses_then_rebuilds_on_changed_text(tmp_path, monkeypatch):
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path)
    emb = FakeEmbedder()
    log = lambda *_: None
    LocalIndex.build_cached("k", _docs("aaaa", "bbbb"), emb, log=log)
    assert emb.doc_calls == 1
    meta = json.loads((tmp_path / "k" / "meta.json").read_text())
    assert len(meta["sha256"]) == 64
    LocalIndex.build_cached("k", _docs("aaaa", "bbbb"), emb, log=log)
    assert emb.doc_calls == 1  # cache hit
    # same chunk count, one changed text: must rebuild
    LocalIndex.build_cached("k", _docs("aaaa", "bbbc"), emb, log=log)
    assert emb.doc_calls == 2


def test_build_cached_rebuilds_on_other_embedder_name(tmp_path, monkeypatch):
    monkeypatch.setattr(index_mod, "INDEX_DIR", tmp_path)
    a, b = FakeEmbedder(), FakeEmbedder()
    b.name = "other"
    log = lambda *_: None
    LocalIndex.build_cached("k", _docs("aaaa"), a, log=log)
    LocalIndex.build_cached("k", _docs("aaaa"), b, log=log)
    assert b.doc_calls == 1
