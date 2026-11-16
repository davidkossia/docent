"""Load the PDFs and split them into sections using each book's PDF outline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from .corpus import BOOKS, Book

FOOTER_RES = [
    re.compile(r"^\d+\s{2,}.+ • .+$"),  # even pages: "52     2 • Expressions"
    re.compile(r"^.+ • .+\s{2,}\d+\s*$"),  # odd pages: "2.5 • Dividing integers     51"
    re.compile(r"^Access for free at openstax\.org\s*$"),
]


@dataclass
class Section:
    id: str
    book: str
    book_title: str
    title: str
    chapter: int
    # (1-based PDF page number, text) pairs, in reading order
    pages: list[tuple[int, str]] = field(default_factory=list)

    @property
    def page_start(self) -> int:
        return self.pages[0][0]

    @property
    def page_end(self) -> int:
        return self.pages[-1][0]

    @property
    def text_len(self) -> int:
        return sum(len(t) for _, t in self.pages)


def clean_text(text: str) -> str:
    """Drop running footers and join words hyphenated across line breaks."""
    lines = [ln for ln in text.split("\n") if not any(r.match(ln.strip()) for r in FOOTER_RES)]
    out = "\n".join(ln.rstrip() for ln in lines)
    out = re.sub(r"(\w)-\n(\w)", r"\1\2", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _walk(reader: PdfReader, outline, depth=0, out=None):
    out = [] if out is None else out
    for item in outline:
        if isinstance(item, list):
            _walk(reader, item, depth + 1, out)
        else:
            out.append((depth, item.title.strip(), reader.get_destination_page_number(item)))
    return out


def _section_id(book: Book, chapter: int, title: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)\s", title)
    if m:
        return f"{book.key}-{m.group(1)}.{m.group(2)}"
    slug = "intro" if title.startswith("Introduction") else "review"
    return f"{book.key}-{chapter}.{slug}"


def _find_title(text: str, title: str) -> int:
    if not re.match(r"^\d+\.\d+\s", title):
        return 0
    pattern = r"\s+".join(re.escape(w) for w in title.split())
    m = re.search(rf"^\s*{pattern}", text, re.M) or re.search(pattern, text)
    return m.start() if m else 0


def load_book(book: Book) -> list[Section]:
    reader = PdfReader(str(book.path))
    entries = _walk(reader, reader.outline)
    # Mark each entry as a section (inside a numbered chapter) or as a boundary only.
    marked: list[tuple[str, int, Section | None]] = []
    chapter = None
    for depth, title, page in entries:
        m = re.match(r"^Chapter (\d+)\b", title)
        if depth == 0:
            chapter = int(m.group(1)) if m else None
            marked.append((title, page, None))
        elif chapter is None or title in ("Chapter Outline", "About OpenStax"):
            marked.append((title, page, None))
        else:
            sec = Section(_section_id(book, chapter, title), book.key, book.title, title, chapter)
            marked.append((title, page, sec))

    by_page: dict[int, list[tuple[str, Section | None]]] = {}
    for title, page, sec in marked:
        by_page.setdefault(page, []).append((title, sec))

    current: Section | None = None
    for p in range(len(reader.pages)):
        text = reader.pages[p].extract_text() or ""
        starters = by_page.get(p, [])
        if not starters:
            if current is not None:
                current.pages.append((p + 1, text))
            continue
        cuts = sorted(((_find_title(text, t), i) for i, (t, _) in enumerate(starters)))
        first = cuts[0][0]
        if current is not None and text[:first].strip():
            current.pages.append((p + 1, text[:first]))
        for k, (pos, i) in enumerate(cuts):
            end = cuts[k + 1][0] if k + 1 < len(cuts) else len(text)
            current = starters[i][1]
            if current is not None and text[pos:end].strip():
                current.pages.append((p + 1, text[pos:end]))
    seen: dict[str, Section] = {}
    for _, _, sec in marked:
        if sec is not None and sec.pages:
            seen.setdefault(sec.id, sec)
    return list(seen.values())


def load_sections() -> list[Section]:
    sections: list[Section] = []
    for book in BOOKS:
        sections.extend(load_book(book))
    return sections
