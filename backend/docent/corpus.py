"""The corpus: two OpenStax textbooks (CC BY-NC-SA 4.0). Downloaded, never committed."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass

from .config import RAW_DIR

BASE = "https://assets.openstax.org/oscms-prodcms/media/documents/"


@dataclass(frozen=True)
class Book:
    key: str
    title: str
    filename: str

    @property
    def url(self) -> str:
        return BASE + self.filename

    @property
    def path(self):
        return RAW_DIR / self.filename


BOOKS = [
    Book("py", "Introduction to Python Programming", "Introduction_to_Python_Programming-WEB.pdf"),
    Book("cs", "Introduction to Computer Science", "Introduction_To_Computer_Science_-_WEB.pdf"),
]
BOOK_BY_KEY = {b.key: b for b in BOOKS}


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for book in BOOKS:
        if book.path.exists():
            print(f"have {book.filename}")
            continue
        print(f"downloading {book.url}")
        urllib.request.urlretrieve(book.url, book.path)  # noqa: S310 (fixed https URL)
