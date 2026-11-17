# Coauthor reproduction guide (PLOS ONE manuscript, November 2026)

This **public GitHub repository** contains the Python analysis scripts, derived analysis inputs, recorded random seeds, generated numerical tables, and stored results supporting the manuscript *Estimating the incidence density rate ratio from a prevalence study using the time since disease onset: identification, estimators and bounds*. No GitHub sign-in is necessary to read or download these public files.

**Use the precise commit linked in the manuscript's Data Availability statement**, rather than relying on whatever changes may later appear on `main`.

## Obtain the code and stored inputs
```sh
git clone https://github.com/ChongWangStat/prevalence-idrr-onset.git
cd prevalence-idrr-onset
# Optionally: git checkout <full-commit-SHA-from-manuscript>
python -m pip install -r requirements-lock.txt
```

Without installing Git, choose **Code → Download ZIP** on the public GitHub repository page.

## Check the existing numerical results without running new simulations
```sh
python check_against_stored.py
sh regenerate_from_stored_results.sh
```
This regenerates numerical tables, derived quantities and figures from the stored outputs, and verifies the stored checksums. It also runs targeted theory checks. This is the practical first check for a coauthor.

## Reproduce all simulation outputs from fixed random seeds
```sh
OFFLINE=1 sh reproduce_all_python.sh
```
This reruns the original Aims 1–3 and added simulation suites and regenerates the two original public-data illustrations from stored derived inputs. The individual-history simulation is computationally expensive (previously several hours). The `OFFLINE=1` mode uses archived NORDCAN extracts and colon-trial aggregates. For a fresh trial-level bootstrap, obtain `colon.csv` as described below and set `COLON_CSV=/full/path/colon.csv`.

## Reproduce the BRFSS–USCS external comparison from the original public data

The following **public raw CDC archives are NOT stored on GitHub** (in contrast to the scripts and the exact expected numerical outputs); obtain them from CDC:

- BRFSS 2022 combined ASCII: https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022ASC.zip
- USCS 1999–2023 ASCII tables (includes the 2022 rows): https://www.cdc.gov/cancer/uscs/USCS-1999-2023-ASCII.zip

Extract `LLCP2022.ASC ` (its archived filename has a trailing space), `BYAGE.TXT`, and `BYAREA.TXT`. Source archive and extracted-file **SHA-256 checksums** are given in [code/brfss_uscs/README_verify.md](code/brfss_uscs/README_verify.md).

Run, from the repository root:
```sh
sh verify_brfss_uscs.sh "/full/path/LLCP2022.ASC " /full/path/BYAGE.TXT /full/path/BYAREA.TXT
```
The wrapper executes the primary analysis and two independently implemented checks and compares them with the recorded text outputs in `results/brfss_uscs/`. The CDC raw records are about 914 MB uncompressed; make sure sufficient disk space is available. These analyses are a descriptive survey-versus-registry comparison, not a validation of the stationary model.

## Reproduce the original real-data illustrations from public sources

- **NORDCAN**: derived Denmark registry extracts, survival summaries and age-specific series are committed under `data/`. Run `python code/python_nordcan_application.py` to use the stored extract; `--download` independently retrieves the specified series from the NORDCAN interface and tests equality (subject to availability/version changes). Sources: https://nordcan.iarc.fr/ and [DATA_SOURCES.md](DATA_SOURCES.md).
- **Colon-cancer trial**: the de-identified `colon.csv` file is distributed with R's `survival` package and publicly mirrored at https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv. It is NOT copied into the repository. Use the `COLON_CSV` option described above for the fresh trial-level bootstrap; the archived `data/colon_verified_aggregates.csv` permits offline table verification.

## Where to look

- `code/`: scripts, including `code/brfss_uscs/`
- `results/python/`, `results/brfss_uscs/`, `results/simultaneous_windows/`: archived analysis results
- `data/`: derived input extracts and verified trial aggregates
- `tex/`: generated numerical macros and table bodies
- `STORED_OUTPUTS_SHA256.txt`, `check_against_stored.py`: numerical integrity checking
- `requirements-lock.txt`: versions of Python packages used for the analyses

The editable manuscript and published-quality TIFF figures are distributed separately in the coauthor's **LaTeX submission package**, not as GitHub analysis files.

## Interpretation and verification limits

The stored numerical results were checked for internal reproducibility. Reproducing a script's output does not independently establish that every epidemiological definition or scientific conclusion is correct. Coauthors should review the original study/registry eligibility criteria, BRFSS questionnaire definitions, and the bound assumptions. The full multi-hour simulations were not rerun at every manuscript-editing step.

For questions about raw public sources and hashes, consult [DATA_SOURCES.md](DATA_SOURCES.md) and [code/brfss_uscs/README_verify.md](code/brfss_uscs/README_verify.md).
