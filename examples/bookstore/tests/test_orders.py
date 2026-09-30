import pytest

from bookstore.cart import Cart
from bookstore.orders import OutOfStock, place_order
from tests.data import catalog


def test_can_buy_the_last_copy():
    cart = Cart(catalog())
    cart.add("111")
    stock = {"111": 1}
    place_order(cart, stock)
    assert stock == {"111": 0}


def test_short_stock_changes_nothing():
    cart = Cart(catalog())
    cart.add("111", 2)
    cart.add("222")
    stock = {"111": 5, "222": 0}
    with pytest.raises(OutOfStock):
        place_order(cart, stock)
    assert stock == {"111": 5, "222": 0}


def test_free_shipping_uses_amount_after_coupon():
    cart = Cart(catalog())
    cart.add("444")                     # 520.00, 1.5 kg
    cart.apply_coupon("SAVE10")         # -52.00 -> 468.00, below 499
    assert place_order(cart, {"444": 3}) == {"books": 468.0, "shipping": 65, "total": 533.0}


def test_repeat_adds_share_one_stock_check():
    cart = Cart(catalog())
    cart.add("333", 2)
    cart.add("333", 2)                  # 4 copies wanted, only 3 in stock
    with pytest.raises(OutOfStock):
        place_order(cart, {"333": 3})
