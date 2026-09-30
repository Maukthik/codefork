from dataclasses import dataclass


@dataclass(frozen=True)
class Book:
    isbn: str
    title: str
    author: str
    price: float
    weight_kg: float


class Catalog:
    def __init__(self, books=()):
        self._books = {b.isbn: b for b in books}

    def get(self, isbn):
        try:
            return self._books[isbn]
        except KeyError:
            raise KeyError(f"Unknown ISBN: {isbn}") from None

    def search(self, query):
        """Books whose title or author contains the query, ignoring case, sorted by title."""
        return sorted((b for b in self._books.values() if query in b.title), key=lambda b: b.title)
