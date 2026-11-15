# Prevalence-survey onset-time analysis: IDRR estimators and bounds

Reproducibility code for the manuscript **Estimating the incidence density rate ratio from a prevalence study using the time since disease onset: identification, estimators and bounds** (Ying Zheng, Chong Wang and Annette O'Connor), prepared for submission to *PLOS ONE* (November 2026).

This repository is for the **analyses and code** and does not include the manuscript's LaTeX or a patient-level clinical dataset. Scripts reproduce simulation results, estimators, confidence intervals, sensitivity bounds, tables, numeric macros and figures from documented random seeds. Reported results and computational validation output are included in `results/` for comparison.

## Installation

Python 3.11+ is required; the manuscript was calculated in Python 3.13. Install dependencies using:

```sh
python -m pip install -r requirements-python.txt
```

## Quick numerical verification

```sh
python code/python_verify_theory.py
python code/python_verify_exact_ci.py
python code/python_check_interval.py
python code/python_falsify_propositions.py
```

After the public registry extracts have been downloaded locally (see below), run `python code/python_validate_outputs.py` for the full 20-dataset audit.

These checks use stored numerical outputs where applicable and **do not** independently prove the mathematical results. The full simulation is computationally intensive (many hours).

## Reproducing figures, analyses and tables

The NORDCAN data are **not redistributed here** while permission to deposit the curated extract is being checked. The publisher makes the underlying aggregate series publicly available. Fetch the exact selected series from the source and regenerate the analysis-input CSV files locally:

```sh
python code/python_nordcan_application.py --download --write-extract
```

Run the complete simulation and analysis sequence from the repository root:

```sh
sh reproduce_all_python.sh
```

The script regenerates numerical results, figure files and LaTeX numeric/table components. It does **not** compile manuscript PDFs, since the manuscript LaTeX is not part of this code repository. The reference NORDCAN release was version **9.6.1**, accessed 13 November 2026. If NORDCAN changes its series, the script's numerical consistency checks may stop: use the versioned source and compare against the stored published results.

The colon-cancer example uses the publicly distributed `survival::colon` dataset; original patient-level data are downloaded by the Python scripts and are **not** redistributed in this repository. Using `OFFLINE=1 sh reproduce_all_python.sh` is supported only after NORDCAN extracts have been fetched and local results have been prepared.

## Structure

- `code/`: Python simulation, analytic, visualization and diagnostic scripts, including reusable survey interval calculations
- `results/`: seeded simulation summaries, tables, diagnostic outputs and provenance metadata
- `data/`: the verified **aggregate** colon-cancer analysis inputs; no patient-level data and no raw NORDCAN extracts
- `examples/`: simulated survey example for the interval-bound calculator
- `figures/`: generated figure outputs, for cross-checking against the manuscript
- `tex/`: generated numeric macros and table bodies, not the manuscript source
- `PYTHON_VALIDATION_REPORT.json`: checks of numerical output consistency
- `DATA_SOURCES.md`: underlying public sources, retrieval notes and redistribution caveats

## Example analysis on grouped survey data

```sh
python code/13_apply_to_survey.py --input examples/simulated_survey_counts.csv --delta0 0.005 --delta1 0.005 --output my_intervals.csv
```

These code files are released as part of an academic reproducibility package. Please cite the published article when available. A formal open-source license should be agreed by the authors before public release; no license grant is asserted by this prepublication preparation archive.