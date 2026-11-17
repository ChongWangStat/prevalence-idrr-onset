# BRFSS 2022 against USCS: files for an independent rerun

Everything needed to rerun and check the survey-against-registry comparison (men against women, all cancers
except non-melanoma skin cancer). The original analysis was written and run with AI assistance (Claude); its skin-only case-definition inconsistency was corrected during subsequent independent review on 16 November 2026.
It has not been rerun by an author. Until someone other than that tool has done so, treat every number as unverified.

## Files

| File | What it is |
|---|---|
| `BRFSS_USCS_analysis_plan.md` | the plan: records, definitions, estimators, the three comparisons, how the result is read |
| `POR_brfss_uscs_validation.py` | the planned analysis, followed by one exploratory analysis (marked in the output) |
| `POR_brfss_uscs_checks.py` | three checks added later: benchmark aligned to the survey estimand, survey design, onset window; ROR/POR with its interval |
| `POR_brfss_uscs_crosscheck.py` | second implementation of the survey estimates: different file reader, jackknife standard errors, no shared code |
| `expected_output_*.txt` | the output of each script as obtained on 16 November 2026 |

## Inputs (public, CDC; not included)

| File | SHA-256 |
|---|---|
| `LLCP2022ASC.zip` (BRFSS 2022 combined landline and cell phone data, from the BRFSS 2022 annual data page) | `ce4e8453afd8685977a13e56da85418eff83e628547604b00e3463eb60c2fc74` |
| unpacked `LLCP2022.ASC ` (914 301 128 bytes; the name inside the zip ends with a space) | `4d3a8b451c4881f5973c9c59be89171beaabccbb7fcfdeaf9f63b3bc50dde52d` |
| `USCS-1999-2023-ASCII.zip` (U.S. Cancer Statistics public-use ASCII files) | `291765c997e59cddc7bc9bdd5852ea18f52696a58e2c212f5eb6b72c680b6df7` |
| unpacked `BYAGE.TXT` | `4b612a23183c13cdf82e964340de8bc95a5ea7f870c43e37706bb1b923fbd081` |
| unpacked `BYAREA.TXT` | `c66cd3df906800b4c75ae5836ce0d0f6497a06af4c7d3da19e93d08407e5221a` |

## Rerun

Python 3 with numpy and scipy (run here with Python 3.13.18, numpy 2.5.3, scipy 1.18.1). Under one minute in all.

```
python3 POR_brfss_uscs_validation.py "LLCP2022.ASC " BYAGE.TXT BYAREA.TXT > out_validation.txt
python3 POR_brfss_uscs_checks.py     "LLCP2022.ASC " BYAGE.TXT            > out_checks.txt
python3 POR_brfss_uscs_crosscheck.py "LLCP2022.ASC "                       > out_crosscheck.txt
diff out_validation.txt ../../results/brfss_uscs/expected_output_validation.txt
diff out_checks.txt     ../../results/brfss_uscs/expected_output_checks.txt
diff out_crosscheck.txt ../../results/brfss_uscs/expected_output_crosscheck.txt
```

The three `diff` commands should print nothing.

## What a rerun shows, and what it does not

A rerun shows that the scripts give the reported numbers from the public files. It does not show that the
choices in the scripts are right. These are the choices to check by reading, with where to check them:

1. **Column positions** (`COLS` at the top of the first two scripts; `field(start, length)` calls in the third)
   against CDC's 2022 variable layout. Variables used: `_STATE`, `DISPCODE`, `_PSU`, `CHCOCNC1`, `CNCRDIFF`,
   `CNCRAGE`, `CNCRTYP2`, `QSTVER`, `_STSTR`, `_LLCPWT`, `_SEX`, `_AGE80`.
2. **States and weight.** The 22 states are those for which CDC's table of 2022 modules lists "Cancer
   Survivorship: Type of Cancer" under the combined data set with `_LLCPWT`. Seven further states asked the
   module on one questionnaire version and are in the version files, which are not used.
3. **Who is kept.** Completed interviews only; questionnaire version 20 dropped in six states. The reasons are
   in the plan, and check 2 of the checks script shows the effect on the sex odds ratio (factor 0.997).
4. **Definitions.** Healthy and case from `CHCOCNC1`; recent from current age minus `CNCRAGE`; the
   exclusion of respondents whose only cancer is a skin cancer of codes 22 or 23.
5. **Registry selection.** Incidence, all sites combined, all races, 2022, ages 20 to 79 in five-year groups.

## Which numbers are which

| Analysis | Status | Where in the output |
|---|---|---|
| ROR, told since last birthday (difference 0), ages 20–79 | planned, primary | validation, each age block |
| The same for ages 20–49 and 50–79 | planned | validation |
| ROR with one known cancer type; ROR under two years (difference 0 or 1) | planned, secondary | validation, lines "S1" and "S2" |
| Unweighted estimates with exact conditional intervals | planned, secondary | validation, end of each ROR line |
| Age-stratified Mantel–Haenszel comparison | exploratory, added after the planned results were seen | validation, last block |
| Aligned benchmark, design checks, window counts, ROR/POR | added after the planned results were seen | checks |

The plan is dated by its file only (16 November 2026, 03:25, one minute before the analysis script). Nothing
outside these files certifies that it preceded the results, so describe the analysis as planned by the
analyst, not as preregistered.

## Results to expect (men against women; survey-weighted; 95% intervals)

| Ages | Registry IDRR, planned / aligned | POR | ROR, told since last birthday | ROR, under two years | ROR/POR, under two years |
|---|---|---|---|---|---|
| 20–79 | 1.08 / 1.06 | 0.73 (0.67–0.79) | 0.92 (0.68–1.25) | 1.01 (0.81–1.26) | 1.39 (1.13–1.70) |
| 20–49 | 0.53 / 0.52 | 0.43 (0.35–0.54) | 0.76 (0.35–1.62) | 0.52 (0.30–0.89) | 1.19 (0.73–1.94) |
| 50–79 | 1.25 / 1.24 | 0.85 (0.78–0.92) | 1.04 (0.76–1.43) | 1.27 (1.00–1.62) | 1.51 (1.21–1.88) |

Second implementation: every point estimate is the same. Jackknife interval limits differ from the linearised
ones by at most 0.01 for ages 20–79 and 50–79 (for example 0.68–1.26 against 0.68–1.25), and the intervals are
wider at ages 20–49, where 27 men are recent cases (0.33–1.73 against 0.35–1.62).


### Final-definition correction

The cleaned analysis counts respondents with a single non-melanoma/unknown skin-cancer history in the no-eligible-cancer group, instead of excluding them from the comparison entirely. This correction was made after the initial analysis; scripts, reports, and the prospective manuscript now match. The original file-checksum list remains unchanged.

The top-level S2 wrapper `verify_brfss_uscs.sh` provides the same three comparisons from the S2 root directory.
