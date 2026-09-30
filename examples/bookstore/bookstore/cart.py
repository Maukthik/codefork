from .money import round_money

# coupon code -> (percent off, minimum subtotal)
COUPONS = {"SAVE10": (10, 500), "BIG20": (20, 2000)}


class Cart:
    def __init__(self, catalog):
        self.catalog = catalog
        self.lines = []          # [isbn, qty], one line per book
        self.coupon = None

    def add(self, isbn, qty=1):
        """Add copies of a book. Adding a book that is already in the cart increases its quantity."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.catalog.get(isbn)
        self.lines.append([isbn, qty])

    def quantity(self, isbn):
        return sum(q for i, q in self.lines if i == isbn)

    def subtotal(self):
        return round_money(sum(self.catalog.get(i).price * q for i, q in self.lines))

    def weight(self):
        return sum(self.catalog.get(i).weight_kg * q for i, q in self.lines)

    def apply_coupon(self, code):
        if code not in COUPONS:
            raise ValueError(f"Unknown coupon: {code}")
        self.coupon = code

    def discount(self):
        """A coupon applies when the subtotal is at least its minimum."""
        if not self.coupon:
            return 0.0
        pct, minimum = COUPONS[self.coupon]
        sub = self.subtotal()
        if sub > minimum:
            return round_money(sub * pct / 100)
        return 0.0

    def total(self):
        return round_money(self.subtotal() - self.discount())
