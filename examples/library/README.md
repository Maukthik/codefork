# Library demo: a green repo for build mode

A tiny lending library (books, copies, loans with a 14-day due date). All 5 tests pass.
There is nothing to fix. It's the starting point for **build mode**: describe a feature and
Fork writes the tests first, then the code that makes them pass.

Try this request:

> Add overdue fines. `return_book` should record a fine of ₹5 for every day a book is
> returned after its due date, capped at ₹100 per loan (no fine on or before the due date).
> Add `Library.fines_owed(member)` returning that member's unpaid total, and
> `Library.pay_fine(member, amount)` that reduces it; paying more than is owed raises `ValueError`.

```bash
python scripts/make_demo.py library
python agent_graph.py --repo workspace/library --build "Add overdue fines. ..."
```
