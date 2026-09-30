# Invoice demo: GST calculator with 3 planted bugs

A small Indian GST invoice calculator. Five of the seven tests fail.

| Bug | Where | Symptom |
|---|---|---|
| Unknown categories silently get 18% instead of raising `ValueError` | `gst.gst_rate` | `test_unknown_category_raises` |
| Intrastate tax is charged twice (full CGST **and** full SGST) instead of split in half | `gst.split_gst` | `test_intrastate_split_is_half_each`, `test_simple_line`, `test_invoice_total` |
| Discount is applied after tax instead of before | `invoice.line_total` | `test_discount_before_tax` |

The fixes span two files, so a correct patch has to reason across modules.
Create a runnable copy with `python scripts/make_demo.py invoice`.
