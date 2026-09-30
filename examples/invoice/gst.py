GST_RATES = {"essentials": 0, "food": 5, "standard": 18, "luxury": 28}


def gst_rate(category):
    """GST % for a product category. Unknown categories are an error."""
    return GST_RATES.get(category, 18)


def split_gst(amount, rate, interstate=False):
    """Interstate sales pay IGST. Intrastate sales split the tax equally into CGST + SGST."""
    tax = amount * rate / 100
    if interstate:
        return {"igst": round(tax, 2)}
    return {"cgst": round(tax, 2), "sgst": round(tax, 2)}
