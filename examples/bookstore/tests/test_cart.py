import pytest

from bookstore.cart import Cart
from tests.data import catalog


def test_adding_same_book_twice_merges_lines():
    cart = Cart(catalog())
    cart.add("111")
    cart.add("111", 2)
    assert cart.lines == [["111", 3]]


def test_coupon_applies_at_exact_minimum():
    cart = Cart(catalog())
    cart.add("222", 2)                  # 500.00
    cart.apply_coupon("SAVE10")
    assert cart.total() == 450.0


def test_coupon_below_minimum_does_nothing():
    cart = Cart(catalog())
    cart.add("333")
    cart.apply_coupon("SAVE10")
    assert cart.total() == 199.0


def test_unknown_coupon():
    with pytest.raises(ValueError):
        Cart(catalog()).apply_coupon("FREE100")
