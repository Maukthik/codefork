from bookstore.catalog import Book, Catalog

BOOKS = [
    Book("111", "The Hobbit", "J. R. R. Tolkien", 350.0, 0.6),
    Book("222", "Wings of Fire", "A. P. J. Abdul Kalam", 250.0, 0.4),
    Book("333", "Malgudi Days", "R. K. Narayan", 199.0, 0.3),
    Book("444", "The Lord of the Rings", "J. R. R. Tolkien", 520.0, 1.5),
]


def catalog():
    return Catalog(BOOKS)
