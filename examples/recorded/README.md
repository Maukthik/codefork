# Recorded runs

Real runs that the hosted demo can replay for free (no model calls, no sandbox).

Every run writes `logs/graph_<run_id>_replay.json`. To publish one, copy it here with a
readable name, for example:

```
copy logs\graph_20261001_101500_ab12_replay.json examples\recorded\bookstore_7_bugs.json
```

The file name becomes the label in the app's "Replay a recorded run" list.
