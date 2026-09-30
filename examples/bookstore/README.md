# Bookstore demo: 7 bugs across 5 modules

A small online bookstore package (catalog, cart, coupons, shipping, orders; prices in INR).
10 of 18 tests fail. It's built so that one fix attempt rarely gets everything: the bugs sit
in different files, and some tests only pass once bugs in *other* modules are fixed too.
That's what parallel branches, partial-fix merging and model escalation are for.

| # | Bug | Where |
|---|---|---|
| 1 | Float `round()` instead of half-up rounding (2.675 becomes 2.67) | `money.round_money` |
| 2 | Search is case-sensitive and ignores the author | `catalog.Catalog.search` |
| 3 | Adding the same book twice creates a second line instead of increasing the quantity | `cart.Cart.add` |
| 4 | Coupon needs subtotal *above* the minimum instead of *at least* | `cart.Cart.discount` |
| 5 | Extra kilograms are floored instead of rounded up | `shipping.shipping_cost` |
| 6 | You can't buy the last copy (`>` instead of `>=`) | `orders.place_order` |
| 7 | Free-shipping threshold uses the amount *before* the coupon | `orders.place_order` (cross-module: cart + shipping) |

Create a runnable copy with `python scripts/make_demo.py bookstore`.
