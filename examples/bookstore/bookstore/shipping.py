import math

FREE_SHIPPING_FROM = 499
BASE = 40
PER_EXTRA_KG = 25


def shipping_cost(amount, weight_kg):
    """Free when the amount is at least 499. Otherwise 40 for the first kg plus 25 for
    each extra kg or part of one: 0.5 kg -> 40, 1.2 kg -> 65, 2.0 kg -> 65, 2.1 kg -> 90."""
    if amount >= FREE_SHIPPING_FROM:
        return 0
    extra = max(0, int(weight_kg - 1))
    return BASE + PER_EXTRA_KG * extra
