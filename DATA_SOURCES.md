# Data used in the three empirical comparisons

The two original illustrations reanalyse public data. Neither is a validation of the stationary model; the added BRFSS--USCS comparison is descriptive rather than formal validation.

## Cancer registry statistics (open population)

Source: NORDCAN, version 9.6.1 (9 November 2026), Association of the Nordic Cancer Registries,
https://nordcan.iarc.fr/ ; retrieved 13 November 2026 through the public interface
(`https://canapi.iarc.who.int/api/nordcan/v2/961`). Cite as in the reference list of the article
(Larønningen et al. 2026; Engholm et al. 2010).

Stored here (only the series used in the article, Denmark, men and women; cancers: lung C33-34,
pancreas C25, colon and rectum C18-20, melanoma of skin C43):

| File | Content |
|---|---|
| `data/nordcan_denmark_extract.csv` | 120 rows: new cases and person-years by calendar year 2015-2024; persons alive on 31 December 2024 diagnosed within 1, 3, 5 and 10 years and at any time, with the population on that date. Each row names the interface endpoint it came from. |
| `data/nordcan_denmark_age.csv` | 864 rows, year 2024, by five-year age group: new cases and person-years (published); prevalence proportions per 100 000 (published) with the numbers of prevalent cases and of residents on 31 December recovered from them (see below) |
| `data/nordcan_denmark_survival.csv` | 8 rows: one-year relative survival, patients diagnosed 2020-2024 |
| `data/nordcan_denmark_extract.json` | source, version and time of retrieval |

`python code/python_nordcan_application.py --download` retrieves the same 56 series again and stops
if any number differs from the stored extract (for example after a new NORDCAN version);
`--write-extract` rebuilds the three data files from the interface, and `--cache DIR` keeps the raw
responses so that the interface is contacted once.

Age-specific counts. NORDCAN publishes the prevalence proportion in each age group, not the numbers
behind it. For each sex and age group the number of residents on 31 December is recovered as the
only integer near the person-years of the year for which every published proportion (four cancers,
five windows) corresponds to a whole number of persons. The script stops if this integer is not
unique or if the recovered counts do not add up to a published total; both checks pass for all 40
prevalence series. In the two youngest age groups of women no prevalent case of these cancers is
recorded, and the residents are allocated in proportion to the person-years.

Terms of use: NORDCAN data may be used freely with the citation above. Systematic retrieval to build
a database requires written permission from the ANCR secretariat; the extract here is limited to the
series analysed (120 totals, 8 survival figures and the age-specific values of 2024 for the same
series).

## Colon-cancer trial (closed cohort)

Source: Moertel et al., Ann Intern Med 1995;122:321-326, doi 10.7326/0003-4819-122-5-199503010-00001.
The 929-patient `colon` dataset is distributed with the R package `survival` and mirrored at
`https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv` (repository
`vincentarelbundock/Rdatasets`, Git blob SHA-1 `aa9f8ebcc4997da14d08a60fed0b32ec2996a9d8`;
1858 rows, two per patient).

The patient-level file is not redistributed in this archive. The scripts read it from the public
source (`--download`) or from a local copy (`--input colon.csv`):

| Script | Output |
|---|---|
| `code/python_colon_bounds.py` | snapshot one year after randomization: recent-onset odds ratio, intervals from the bounds, tipping point, survival after recurrence (`results/python/colon_bounds.csv`, `colon_postrecurrence_km.csv`) |
| `code/python_colon_trial_application.py` | 15 snapshot-by-window aggregates, checked against `data/colon_verified_aggregates.csv` |
| `code/python_colon_bootstrap.py` | patient-level bootstrap (4000 replicates, seed 20261112); `results/python/colon_bootstrap_provenance.json` records the checksum of the file analysed |

On 13 November 2026 all three were run on the public file; the bootstrap output was identical to the
stored `results/python/colon_bootstrap_ci.csv`. Without network access the tables and the figure are
rebuilt from the stored aggregate results (`--from-results`, `--verified-aggregates`).


## Added independent external comparison (BRFSS 2022 + USCS 2022)

- **BRFSS**: CDC, 2022 Behavioral Risk Factor Surveillance System combined landline/cell public-use ASCII records, https://www.cdc.gov/brfss/annual_data/annual_2022.html . Public records, only states with the Cancer Survivorship: Type of Cancer module in the combined file.
- **USCS**: CDC/NCI, United States Cancer Statistics, 2025 submission (incidence through 2023, released June 2026), https://www.cdc.gov/united-states-cancer-statistics/dataviz/data-tables.html .
- Exact original-source archive and extracted-file SHA-256 hashes are documented in `code/brfss_uscs/README_verify.md`. This S2 ZIP contains the code and results, **not** the two raw CDC archives.
- Note: cancer histories, first cancer versus subsequent registry primaries, and survey participation/survival are not directly exchangeable; this comparison cannot establish unbiasedness under all assumptions.
