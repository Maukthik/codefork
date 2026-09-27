# Creates workspace\invoice: a GST invoice calculator with 3 planted bugs.
# Run from the fork folder:  powershell -ExecutionPolicy Bypass -File setup_invoice_demo.ps1
$d = "workspace\invoice"
New-Item -ItemType Directory -Force $d | Out-Null

@'
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
'@ | Set-Content (Join-Path $d "gst.py")

@'
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
'@ | Set-Content (Join-Path $d "invoice.py")

@'
import pytest
from gst import gst_rate, split_gst


def test_known_rates():
    assert gst_rate("food") == 5
    assert gst_rate("luxury") == 28


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        gst_rate("spaceship")


def test_intrastate_split_is_half_each():
    assert split_gst(1000, 18) == {"cgst": 90.0, "sgst": 90.0}


def test_interstate_is_igst():
    assert split_gst(1000, 18, interstate=True) == {"igst": 180.0}
'@ | Set-Content (Join-Path $d "test_gst.py")

@'
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
'@ | Set-Content (Join-Path $d "test_invoice.py")

git -C $d init -q
git -C $d add .
git -C $d -c user.name="Fork Demo" -c user.email="demo@localhost" commit -q -m "GST invoice demo (buggy)"
Write-Host "Created $d. Check it with:  Push-Location $d; python -m pytest -q; Pop-Location"
