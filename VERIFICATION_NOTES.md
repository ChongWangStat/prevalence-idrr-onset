# Verification performed when preparing this GitHub package (15 November 2026)

All runs used the source code as bundled here, under Python 3.13, before repository publication.

- `python code/python_verify_theory.py`: **PASS** (theoretical identities checked numerically).
- `python code/python_verify_exact_ci.py`: **PASS** (12 exact-interval comparisons including zero counts).
- `python code/python_check_interval.py`: **PASS** (early-exit bounds and short-window calculations).
- `python code/python_validate_outputs.py`: **PASS** (20 numerical datasets, 56 scenario groups, 13 interval scenarios and registry/trial checks), after temporarily supplying the originally extracted NORDCAN CSVs. Those original extracted CSV files were removed before the public-ready archive was built.

The computational checks do not substitute for an independent human verification of the mathematical proofs, and a complete rerun of the 10,000-replicate experiments was not performed during this packaging step.

No public repository was created or uploaded by this preparation process.