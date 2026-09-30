from .money import round_money
from .shipping import shipping_cost


class OutOfStock(Exception):
    pass


def place_order(cart, inventory):
    """Check stock for every line, then reduce it, and return the bill.
    If any book is short, raise OutOfStock and change nothing.
    Shipping is free when the amount paid for books (after the coupon) is at least 499."""
    for isbn, qty in cart.lines:
        if not inventory.get(isbn, 0) > qty:
            raise OutOfStock(isbn)
    for isbn, qty in cart.lines:
        inventory[isbn] -= qty
    books = cart.total()
    shipping = shipping_cost(cart.subtotal(), cart.weight())
    return {"books": books, "shipping": shipping, "total": round_money(books + shipping)}
