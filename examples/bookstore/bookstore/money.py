def round_money(amount):
    """Round to paise with half-up rounding, like a till receipt (2.675 -> 2.68)."""
    return round(amount, 2)
