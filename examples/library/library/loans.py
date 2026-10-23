from dataclasses import dataclass
from datetime import date, timedelta

LOAN_DAYS = 14


@dataclass
class Book:
    isbn: str
    title: str
    copies: int


@dataclass
class Loan:
    isbn: str
    member: str
    borrowed: date
    due: date


class Library:
    def __init__(self):
        self.books: dict[str, Book] = {}
        self.loans: list[Loan] = []

    def add_book(self, isbn: str, title: str, copies: int = 1) -> None:
        if copies <= 0:
            raise ValueError("copies must be positive")
        if isbn in self.books:
            self.books[isbn].copies += copies
        else:
            self.books[isbn] = Book(isbn, title, copies)

    def available(self, isbn: str) -> int:
        """Copies of a book on the shelf right now."""
        book = self.books.get(isbn)
        if book is None:
            raise KeyError(f"Unknown ISBN: {isbn}")
        return book.copies - sum(1 for l in self.loans if l.isbn == isbn)

    def checkout(self, isbn: str, member: str, today: date) -> Loan:
        if self.available(isbn) <= 0:
            raise ValueError(f"No copies of {isbn} left")
        if any(l.isbn == isbn and l.member == member for l in self.loans):
            raise ValueError(f"{member} already has {isbn}")
        loan = Loan(isbn, member, today, today + timedelta(days=LOAN_DAYS))
        self.loans.append(loan)
        return loan

    def return_book(self, isbn: str, member: str, today: date) -> Loan:
        for loan in self.loans:
            if loan.isbn == isbn and loan.member == member:
                self.loans.remove(loan)
                return loan
        raise ValueError(f"{member} doesn't have {isbn}")
