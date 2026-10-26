# Phase 1 fixtures

`ingestion.csv` is a small, manually authored **synthetic test fixture**, not an
excerpt of the supplied source data and not a labelled incident dataset. It
contains quoted commas/newlines, tied/out-of-order timestamps, a duplicate,
unknown and empty messages, optional invalid fields, an invalid timestamp, and
a wrong-width row. Expected accounting: 8 parsed, 6 accepted, 2 rejected.

Additional boundary and failure inputs are generated locally by pytest fixtures.
No fixture labels, detection events, or anomaly assumptions are used in Phase 1.
