#!/bin/sh
# Complete numerical reproduction of every result from the recorded seeds, followed by
# the tables and figures. Run from the repository root with the Python environment
# of requirements-python.txt. The individual-history simulations take about an hour on
# four cores, and the individual-history repetition of the estimator comparison several hours
# more; the remaining steps take a few minutes.
#
# OFFLINE=1 avoids network after obtaining the public NORDCAN extract once;
#            the colon-trial tables are rebuilt from stored aggregates.
# NORDCAN extracts are not redistributed; fetch them before first run:
#   python code/python_nordcan_application.py --download --write-extract
set -eu
cd "$(dirname "$0")"
mkdir -p results/python results/simultaneous_windows

# numerical checks of the theory
python code/python_verify_theory.py
python code/python_verify_exact_ci.py
python code/python_check_interval.py
python code/python_falsify_propositions.py

# simulations
python code/python_original_reproduction.py --reps 10000 --monitoring 1000 --out results/python
python code/python_secondary_reproduction.py --validation 5000 --trajectories 2000 --immigration 5000 --out results/python
python code/python_growth_reproduction.py --reps 5000 --out results/python
python code/python_changing_incidence_reproduction.py --reps 5000 --out results/python
python code/python_main_reproduction.py --reps 10000 --out results/python
python code/python_interval_reproduction.py --reps 10000 --out results/python
# the comparison of the estimators repeated with individual histories (about three hours on two cores)
python code/python_main_individual_check.py --reps 10000 --out results/python
python code/python_main_individual_check.py --compare --out results/python
python code/11_simultaneous_windows.py --reps 3000 --workers 4 --output-dir results/simultaneous_windows
python code/12_sensitivity_figure.py --result-dir results/simultaneous_windows --figure-dir figures

# illustrations with real data
if [ "${OFFLINE:-${COLON_OFFLINE:-0}}" = 1 ]; then
  python code/python_nordcan_application.py
  python code/python_colon_trial_application.py --verified-aggregates --outdir results/python
  test -s results/python/colon_bootstrap_ci.csv      # the bootstrap needs the patient-level file
  python code/python_colon_bounds.py --from-results
else
  python code/python_nordcan_application.py --download
  python code/python_colon_trial_application.py --download --outdir results/python
  python code/python_colon_bootstrap.py --download --reps 4000 --seed 20261112 --outdir results/python
  python code/python_colon_bounds.py --download
fi
python code/python_make_colon_table.py
python code/python_make_colon_bootstrap_table.py
cp results/python/colon_bootstrap_comparison.pdf figures/figS3_colon_trial.pdf

# numbers quoted in the text, tables, figures, checks and PDFs
python code/python_all_numbers.py
python code/python_all_tables.py
python code/python_all_figures.py
python code/python_check_targeted_bootstrap.py
python code/python_validate_outputs.py
python code/python_figure_tiffs.py
printf '\nReproduction complete: results, table bodies, numeric macros, and figures regenerated.\n'
