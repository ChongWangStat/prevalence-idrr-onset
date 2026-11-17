#!/bin/sh
# Rebuild the numbers quoted in the text, the table bodies and the figures from the stored
# simulation and analysis results (results/, data/), and compare them with the stored
# outputs of the article. No simulation is rerun and no network access is needed; takes a
# few minutes. For a complete recalculation from the seeds run reproduce_all_python.sh.
set -eu
cd "$(dirname "$0")"
mkdir -p tex figures
python code/python_verify_theory.py
python code/python_verify_exact_ci.py
python code/python_check_interval.py
python code/python_falsify_propositions.py
python code/12_sensitivity_figure.py --result-dir results/simultaneous_windows --figure-dir figures
python code/python_nordcan_application.py
python code/python_main_individual_check.py --compare --out results/python
python code/python_colon_trial_application.py --verified-aggregates --outdir results/python
test -s results/python/colon_bootstrap_ci.csv
python code/python_colon_bounds.py --from-results
python code/python_make_colon_table.py
python code/python_make_colon_bootstrap_table.py
cp results/python/colon_bootstrap_comparison.pdf figures/figS3_colon_trial.pdf
python code/13_apply_to_survey.py --input examples/simulated_survey_counts.csv --delta0 0.005 --delta1 0.005 --output results/simultaneous_windows/cli_application_example.csv
python code/python_all_numbers.py
python code/python_all_tables.py
python code/python_all_figures.py
python code/python_check_targeted_bootstrap.py
python code/python_validate_outputs.py
python code/python_figure_tiffs.py
python check_against_stored.py
