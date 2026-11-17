#!/bin/sh
# Usage: sh verify_brfss_uscs.sh '/full/path/LLCP2022.ASC ' '/full/path/BYAGE.TXT' '/full/path/BYAREA.TXT'
set -eu
cd "$(dirname "$0")"
if [ "$#" -ne 3 ]; then
  echo "usage: sh verify_brfss_uscs.sh <LLCP2022.ASC> <BYAGE.TXT> <BYAREA.TXT>" >&2
  exit 2
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
python code/brfss_uscs/POR_brfss_uscs_validation.py "$1" "$2" "$3" > "$work/validation.txt"
python code/brfss_uscs/POR_brfss_uscs_checks.py "$1" "$2" > "$work/checks.txt"
python code/brfss_uscs/POR_brfss_uscs_crosscheck.py "$1" > "$work/crosscheck.txt"
diff -u results/brfss_uscs/expected_output_validation.txt "$work/validation.txt"
diff -u results/brfss_uscs/expected_output_checks.txt "$work/checks.txt"
diff -u results/brfss_uscs/expected_output_crosscheck.txt "$work/crosscheck.txt"
echo 'PASS: all three BRFSS-USCS outputs match the archived expected results'
