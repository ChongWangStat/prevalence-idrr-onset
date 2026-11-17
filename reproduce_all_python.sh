#!/bin/sh
# Complete numerical reproduction of every result of the article from the recorded seeds:
# numerical checks of the theory, simulations, the two illustrations with real data, the
# numbers quoted in the text (tex/numbers.tex, tex/numbers_app.tex), the table bodies
# (tex/tab_*_body.tex) and the figures (figures/). The last step compares every output with
# the stored outputs of the article (STORED_OUTPUTS_SHA256.txt).
#
# Run with the Python environment of requirements-lock.txt. The individual-history
# simulations take most of the time (several hours on two cores; see README.md).
#
# OFFLINE=1         do not contact the two public data sources: the registry series stored in
#                   data/ are used as they are, and the colon-trial tables are rebuilt from the
#                   stored aggregate results (the patient-level file is not redistributed here).
# COLON_CSV=<file>  absolute path of a local copy of the trial's patient-level file, read
#                   instead of downloading it (source and checksum in DATA_SOURCES.md). The
#                   trial results and the bootstrap are then recomputed, also with OFFLINE=1.
set -eu
cd "$(dirname "$0")"
mkdir -p results/python results/simultaneous_windows tex figures
OFFLINE="${OFFLINE:-${COLON_OFFLINE:-0}}"

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
# the comparison of the estimators repeated with individual histories (the longest step)
python code/python_main_individual_check.py --reps 10000 --out results/python
python code/python_main_individual_check.py --compare --out results/python
python code/11_simultaneous_windows.py --reps 3000 --workers 4 --output-dir results/simultaneous_windows
python code/12_sensitivity_figure.py --result-dir results/simultaneous_windows --figure-dir figures

# illustrations with real data: cancer registry
if [ "$OFFLINE" = 1 ]; then
  python code/python_nordcan_application.py
else
  python code/python_nordcan_application.py --download
fi
# illustrations with real data: colon-cancer trial
if [ -n "${COLON_CSV:-}" ]; then
  python code/python_colon_trial_application.py --input "$COLON_CSV" --outdir results/python
  python code/python_colon_bootstrap.py --input "$COLON_CSV" --reps 4000 --seed 20261112 --outdir results/python
  python code/python_colon_bounds.py --input "$COLON_CSV"
elif [ "$OFFLINE" = 1 ]; then
  python code/python_colon_trial_application.py --verified-aggregates --outdir results/python
  test -s results/python/colon_bootstrap_ci.csv      # the bootstrap needs the patient-level file
  python code/python_colon_bounds.py --from-results
else
  python code/python_colon_trial_application.py --download --outdir results/python
  python code/python_colon_bootstrap.py --download --reps 4000 --seed 20261112 --outdir results/python
  python code/python_colon_bounds.py --download
fi
python code/python_make_colon_table.py
python code/python_make_colon_bootstrap_table.py
cp results/python/colon_bootstrap_comparison.pdf figures/figS3_colon_trial.pdf

# worked example of the tool for a user's own counts (README.md)
python code/13_apply_to_survey.py --input examples/simulated_survey_counts.csv --delta0 0.005 --delta1 0.005 --output results/simultaneous_windows/cli_application_example.csv

# numbers quoted in the text, tables, figures and checks
python code/python_all_numbers.py
python code/python_all_tables.py
python code/python_all_figures.py
python code/python_check_targeted_bootstrap.py
python code/python_validate_outputs.py
python code/python_figure_tiffs.py

# comparison with the stored outputs of the article
python check_against_stored.py
