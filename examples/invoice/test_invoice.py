from invoice import line_total, invoice_total


def test_simple_line():
    r = line_total(100, 2, "standard")
    assert r["base"] == 200
    assert r["total"] == 236.0


def test_discount_before_tax():
    r = line_total(1000, 1, "standard", discount_pct=10)
    assert r["base"] == 900
    assert r["taxes"] == {"cgst": 81.0, "sgst": 81.0}
    assert r["total"] == 1062.0


def test_invoice_total():
    lines = [
        {"price": 50, "qty": 4, "category": "food"},
        {"price": 1000, "qty": 1, "category": "luxury", "interstate": True},
    ]
    assert invoice_total(lines) == 1490.0
