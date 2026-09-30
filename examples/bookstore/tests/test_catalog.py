from tests.data import catalog


def test_search_ignores_case():
    assert [b.isbn for b in catalog().search("hobbit")] == ["111"]


def test_search_matches_author_sorted_by_title():
    assert [b.title for b in catalog().search("tolkien")] == ["The Hobbit", "The Lord of the Rings"]
