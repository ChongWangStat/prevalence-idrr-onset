# Code and data: Estimating the incidence density rate ratio from a prevalence study using the time since disease onset: identification, estimators and bounds

Ying Zheng, Chong Wang, Annette O'Connor. S2 Code and Data of the article submitted to *PLOS ONE*.

This repository contains the Python code, the derived input data, the recorded seeds and the stored outputs
behind every number, table and figure of the article and of S1 Appendix, including the external BRFSS--USCS comparison. It does not contain the
manuscript.

## Contents

| Path | Content |
|---|---|
| `code/` | Python scripts: checks of the theory, simulations, the two original illustrations, numbers, tables, figures, and external survey--registry scripts |
| `data/` | registry extract (NORDCAN, Denmark: totals, age-specific values, survival) and verified aggregates of the trial example |
| `examples/` | example input for the tool that applies the bounds to a user's own counts |
| `results/python/` | stored output of every simulation and analysis (CSV, JSON); `logs/` holds screen output of earlier runs |
| `results/simultaneous_windows/` | stored output of the several-windows simulation and of the worked example |
| `tex/` | generated LaTeX: one body per table (`tab_*_body.tex`) and the numbers quoted in the text (`numbers.tex`, `numbers_app.tex`) |
| `figures/` | source for Figure 1; other figures are regenerated from scripts, while submission TIFFs are in the LaTeX package |
| `DATA_SOURCES.md` | the public data sources and their terms |
| `code/brfss_uscs/`, `results/brfss_uscs/` | independent survey--registry comparison scripts, plan, expected outputs; CDC raw records are downloaded separately |
| `STORED_OUTPUTS_SHA256.txt` | checksums of the stored outputs, used by `check_against_stored.py` |
| `PYTHON_VALIDATION_REPORT.json` | report of `code/python_validate_outputs.py` |

## Coauthor access and complete rerun instructions

For a step-by-step guide, including direct public-source download URLs, original-file checksums, and the differences between quick verification and the full seeded rerun, see **[COAUTHOR_REPRODUCIBILITY.md](COAUTHOR_REPRODUCIBILITY.md)**. The original large CDC archives are not hosted here; the code, derived data, and expected outputs are public. For the exact version corresponding to the manuscript, use the commit linked in its Data Availability statement.

## Requirements

Python 3.11 or later. `pip install -r requirements-lock.txt` installs the versions used for the
article (with Python 3.13.18); `requirements-python.txt` gives the minimum versions. Fig 1 is a
TikZ drawing; recompiling it needs `pdflatex`, nothing else does.

## Running

```
sh regenerate_from_stored_results.sh              # numbers, tables and figures from the stored results
OFFLINE=1 sh reproduce_all_python.sh              # everything from the seeds, without network access
sh reproduce_all_python.sh                        # the same, and the two public data sources are read again
COLON_CSV=/path/to/colon.csv OFFLINE=1 sh reproduce_all_python.sh    # trial file from a local copy
```

To independently rerun the 2022 BRFSS--USCS comparison, download and extract the public CDC files
listed in `code/brfss_uscs/README_verify.md`, then run:

```sh
sh verify_brfss_uscs.sh /path/to/LLCP2022.ASC /path/to/BYAGE.TXT /path/to/BYAREA.TXT
```

This separate check is **not** part of `reproduce_all_python.sh`, because the CDC raw ASCII
archives are not redistributed. The three recorded BRFSS--USCS expected outputs are included.

`python` must be the interpreter of the environment above.

- `regenerate_from_stored_results.sh` reruns the numerical checks of the theory, the two
  illustrations and the scripts that write the numbers, the table bodies and the figures. No
  simulation is rerun. About two minutes.
- `reproduce_all_python.sh` reruns every simulation from its recorded seeds first. On two cores the
  complete run took about 2 hours 45 minutes, of which the repetition of the comparison of the estimators
  with individual histories (`python_main_individual_check.py`) took about 2 hours.
- With `OFFLINE=1` the registry series in `data/` are used as they are, and the trial tables are
  rebuilt from the stored aggregate results, because the patient-level trial file is not
  redistributed here. Without it the registry series are downloaded again and the script stops if
  any number differs from `data/`; the trial file is downloaded, and its checksum and the
  aggregates computed from it are compared with the recorded ones.
  `COLON_CSV` (absolute path) reads the trial file from a local copy instead; the trial results and
  the bootstrap are then recomputed also with `OFFLINE=1`.

Both scripts end with `check_against_stored.py`, which compares every output with the stored output
of the article and stops with an error if a numerical output differs. Every replicate has its own
seed, so the results do not depend on the number of threads.

## Which source files support the current manuscript (v16)

The published manuscript has been condensed. The additional full numerical tables and simulation grids remain in this archive as supporting reproducible outputs; **they are not necessarily numbered tables in the main text**. The complete 12-row estimator comparison table removed from the main text is retained in `documentation/archived_main_estimator_table.tex`.

Main manuscript tables: original simulation summary (`tex/tab_orig_body.tex`), selected interval sensitivity scenarios (`results/python/interval_bounds_python.csv`), NORDCAN cancer registry (`results/python/nordcan_application.csv`) and BRFSS--USCS survey--registry comparison (`results/brfss_uscs/`). Main figures: the model, onset-specific odds ratio, simulation bias and precision, NORDCAN and colon-cancer trial (six TIFFs supplied separately with the manuscript).

S1 Appendix retains supporting proofs, essential parameter settings, the simultaneous-window and colon-cancer calculations, and BRFSS methods. The original two simulation pseudocode algorithms moved out of S1 are in `documentation/simulation_algorithms.tex`; the executable programs remain in `code/`. The full series of generated LaTeX tables and figures are archived in `tex/` and `figures/`; their *legacy numeric labels* correspond to the earlier full-length analysis and should not be substituted for the current paper's table or figure numbers.

## Simulations

| Script | What it simulates | How |
|---|---|---|
| `python_original_reproduction.py` | Aims 1 to 3; stationarity of the simulated populations | individual histories (Algorithm S2) |
| `python_secondary_reproduction.py` | checks of the model and population departures; recent start; immigration of prevalent cases | individual histories |
| `python_growth_reproduction.py`, `python_changing_incidence_reproduction.py` | growth; incidence rates that change over time | individual histories |
| `python_main_reproduction.py` | comparison of the estimators, 56 scenarios | samples the stationary distribution in background result B1 (Algorithm S1) |
| `python_main_individual_check.py` | the same comparison repeated, and compared with the previous line | individual histories |
| `python_interval_reproduction.py` | onset recorded in intervals; bounds | samples the stationary law |
| `11_simultaneous_windows.py` | several windows | samples the stationary law |

The individual-history kernels are in `python_individual_histories.py`.

## Checks

- `python_falsify_propositions.py` looks for counterexamples to each proposition among duration
  distributions outside those of the simulation study.
- `python_verify_theory.py`, `python_verify_exact_ci.py` and `python_check_interval.py` check
  identities, the exact confidence limits (against SciPy) and the statements supporting the early-exit identification and sensitivity-bound proposition.
- `python_validate_outputs.py` checks row counts, ranges and consistency of all outputs and writes
  `PYTHON_VALIDATION_REPORT.json`; `python_check_targeted_bootstrap.py` checks the trial bootstrap.
- `check_against_stored.py` compares all outputs with `STORED_OUTPUTS_SHA256.txt`.

## Applying the bounds to your own counts

```
python code/13_apply_to_survey.py --input examples/simulated_survey_counts.csv --delta0 0.005 --delta1 0.005 --output my_intervals.csv
```

The input has one row per exposure group with the number of healthy individuals and the numbers of
cases by interval of time since onset; `--delta0` and `--delta1` are the limits of Equation 13 per
day. The stored result of this command is `results/simultaneous_windows/cli_application_example.csv`.

## Notes

- The stored results in `results/` are the output of one uninterrupted run of
  `reproduce_all_python.sh`. The option `--resume` of the three longest simulations reads the
  rows already written and writes them again, which cuts small values at the 16th decimal place
  (relative differences below 1e-12). After a resumed run `check_against_stored.py` therefore
  reports those files as different although the tables, numbers and figures are the same.
- In `tex/numbers.tex` the macros that begin with `nRef`, and `nWordCount`, are cross-reference
  numbers and a word count from the build of the manuscript. They are not results;
  `python_all_numbers.py` keeps them as they are and requires the file to exist.
- `tex/tab_por_body.tex`, `tab_orig_same_body.tex` and `tab_orig_diff_body.tex` are written by
  `python_all_tables.py` but are not used in the article.
- `code/python_merge_and_audit.py` compared an earlier run of the main simulation with results from
  R scripts of an earlier version (`results/python/main_python_audit.json`, `main_audit_log.txt`).
  Those R results are not part of this archive, so the script cannot be rerun here; no number in
  the article comes from it.


## External survey--registry comparison (main paper and S1 Appendix)

The 2022 BRFSS survey was compared with 2022 USCS incidence counts. This is an external **comparison**, not an independent validation of the full theoretical model. The primary recent-onset definition uses the difference of zero completed years between respondent age and age at first diagnosis; the secondary definition uses differences of zero or one year. The latter is an approximation to a recent-onset window, not an observed diagnosis date.

All survey scripts, the original analysis plan, the post-analysis harmonization notice, and the results are in `code/brfss_uscs/` and `results/brfss_uscs/`. Public CDC source archives are *not* redistributed: download them from the URLs in `code/brfss_uscs/README_verify.md` and check their SHA-256 digests. After extraction, run the three commands documented in that README. Skin-only histories of non-melanoma/unknown type were excluded from eligible-cancer numerators and included in the no-eligible-cancer denominator **post hoc**, consistently for both POR and ROR; no claim of prospective registration is made.

### Targeted trial-data correction in this package

`code/python_colon_bounds.py` now evaluates the Kaplan--Meier excess-survival envelope at actual event times and their left limits, instead of a five-day grid. The observed maximum changes from approximately 2.53 to **2.90 per 1,000 days** (52 days after recurrence) for each analysed window; the 365-day tipping point remains 2.64 per 1,000 days. `results/python/colon_bounds.csv`, associated number macros, table and Figure 6 were updated. The original stored simulation outputs are otherwise unchanged.


From the S2 root, run the external comparison against the downloaded public CDC files with:

```sh
sh verify_brfss_uscs.sh '/path/LLCP2022.ASC ' '/path/BYAGE.TXT' '/path/BYAREA.TXT'
```

The shell script compares all three result files against the stored outputs (including the independent jackknife cross-check). It does **not** run by default during the other paper analyses, because the raw CDC files are not redistributed.
