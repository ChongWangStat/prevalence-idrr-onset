# Data sources and redistribution

## NORDCAN aggregate cancer-registry statistics

Source: NORDCAN, version 9.6.1 (9 November 2026), Association of the Nordic Cancer Registries, https://nordcan.iarc.fr/ ; analysis series accessed 13 November 2026 via the public interface `https://canapi.iarc.who.int/api/nordcan/v2/961`.

This reproduction uses the 2015-2024 annual counts/person-years, 2024 prevalence by time since diagnosis, age-specific prevalence summaries, and one-year relative survival for selected Denmark cancers by sex. The NORDCAN site says cancer tables and figures may be used and published with citation, but the original project's data documentation asks authors to confirm permission before placing the curated machine-readable NORDCAN extract into a public archive. For caution, `data/nordcan_denmark_*` files are excluded here. Run `python code/python_nordcan_application.py --download --write-extract` to retrieve the same source series. Cite NORDCAN as in the article.

The stored `results/python/nordcan*` files contain published **analysis outputs**, not the original registry extract, and permit checking the figures/numeric claims without downloading the registry series. They do not replace public access to the underlying registry statistics.

## Colon-cancer trial

The historical 929-patient trial data are distributed in R's `survival` package and at https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv . The code retrieves the public patient-level file; it is not redistributed here. The small `data/colon_verified_aggregates.csv` file contains study-derived aggregate counts used for cross-checking.

## Simulation data

The model parameters, simulated results, seeds, and software environment are documented in `README.md`, `requirements-python.txt`, the scripts and `results/`.