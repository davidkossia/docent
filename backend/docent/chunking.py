"""Chunking strategies. Each strategy turns sections into LangChain Documents."""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingest import Section, clean_text

MIN_OVERLAP_CHARS = 100  # a chunk counts as "covering" a section if it holds this much of it


@dataclass(frozen=True)
class Strategy:
    name: str
    chunk_size: int
    overlap: int
    clean: bool  # strip footers, join hyphenated words
    by_section: bool  # split inside sections, never across them
    header: bool  # prepend "Book - Section" to each chunk
    description: str


STRATEGIES = {
    s.name: s
    for s in [
        Strategy("fixed-1000", 1000, 200, False, False, False,
                 "Raw page text, 1,000-character windows, 200 overlap, ignores section boundaries"),
        Strategy("fixed-400", 400, 60, False, False, False,
                 "Raw page text, 400-character windows, 60 overlap, ignores section boundaries"),
        Strategy("section-800", 800, 100, True, True, True,
                 "Cleaned text, split inside sections, 800 characters, 100 overlap, section title prepended"),
        Strategy("section-500", 500, 80, True, True, True,
                 "Cleaned text, split inside sections, 500 characters, 80 overlap, section title prepended"),
    ]
}


@dataclass
class _Seg:
    start: int
    end: int
    page: int
    section: Section


def _split(strategy: Strategy, sections: list[Section], book_title: str | None) -> list[Document]:
    """Split one text stream made from `sections` (in order) and tag each chunk."""
    parts, segs, pos = [], [], 0
    for sec in sections:
        for page, text in sec.pages:
            text = clean_text(text) if strategy.clean else text
            if not text.strip():
                continue
            segs.append(_Seg(pos, pos + len(text), page, sec))
            parts.append(text)
            pos += len(text) + 1  # joined by "\n"
    stream = "\n".join(parts)
    starts = [s.start for s in segs]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=strategy.chunk_size, chunk_overlap=strategy.overlap, add_start_index=True
    )
    docs: list[Document] = []
    for d in splitter.create_documents([stream]):
        a = d.metadata["start_index"]
        b = a + len(d.page_content)
        first = max(0, bisect.bisect_right(starts, a) - 1)
        last = max(0, bisect.bisect_right(starts, b - 1) - 1)
        covered: dict[str, int] = {}
        for seg in segs[first : last + 1]:
            overlap = min(b, seg.end) - max(a, seg.start)
            if overlap > 0:
                covered[seg.section.id] = covered.get(seg.section.id, 0) + overlap
        main = segs[first].section
        ids = [main.id] + [
            sid for sid, n in covered.items() if sid != main.id and n >= MIN_OVERLAP_CHARS
        ]
        text = d.page_content
        if strategy.header:
            text = f"{main.book_title} | {main.title}\n{text}"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "book": main.book,
                    "book_title": main.book_title,
                    "section_id": main.id,
                    "section_title": main.title,
                    "section_ids": ids,
                    "page_start": segs[first].page,
                    "page_end": segs[last].page,
                },
            )
        )
    return docs


def build_chunks(strategy: Strategy, sections: list[Section]) -> list[Document]:
    docs: list[Document] = []
    if strategy.by_section:
        for sec in sections:
            docs.extend(_split(strategy, [sec], sec.book_title))
    else:
        for book in dict.fromkeys(s.book for s in sections):
            docs.extend(_split(strategy, [s for s in sections if s.book == book], None))
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = f"{strategy.name}:{i}"
    return docs
