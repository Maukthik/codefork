from gst import gst_rate, split_gst


def line_total(price, qty, category, discount_pct=0, interstate=False):
    """Discount applies to the base amount BEFORE tax is calculated."""
    base = price * qty
    rate = gst_rate(category)
    taxes = split_gst(base, rate, interstate)
    total = base + sum(taxes.values())
    total -= total * discount_pct / 100
    return {"base": round(base, 2), "taxes": taxes, "total": round(total, 2)}


def invoice_total(lines):
    return round(sum(line_total(**line)["total"] for line in lines), 2)
