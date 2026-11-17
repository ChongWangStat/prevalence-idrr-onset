# Pre-publication verification, 17 November 2026

- The manuscript main/S1 consistency audit is in the submission ZIP's `V16_CONSISTENCY_REPORT.md`.
- `python code/python_validate_outputs.py`: PASS (20 numeric datasets; 56 scenario groups; 13 interval scenarios).
- `python check_against_stored.py`: PASS (88/88 stored output files identical).
- `sh verify_brfss_uscs.sh` against the CDC's source archives: PASS (all three expected output text files identical); see `code/brfss_uscs/README_verify.md` for exact source checksums.
- `python code/python_verify_theory.py`: PASS (the numerical identity checks); mathematical proof review by human coauthors remains separate.
- The complete multi-hour original Monte Carlo rerun was not repeated during this consistency audit.
- Exact colon-cancer code, derived numbers, and all BRFSS--USCS scripts in this archive supersede the original repository snapshot from 15 November 2026.
- Original third-party raw BRFSS and USCS files are not redistributed; download instructions and hashes are supplied.
