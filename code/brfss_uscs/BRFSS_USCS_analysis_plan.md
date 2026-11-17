# BRFSS 2022 against USCS: analysis plan, fixed before any sex-specific result was computed

Written 16 November 2026, after the structural checks only (column positions, which states and
questionnaire versions carried the cancer module, module completion by interview disposition, sexes pooled).
No count, weight total or ratio by sex had been looked at when this was written.

## Question
Does the recent-onset odds ratio of the manuscript, computed from one independent population survey,
agree with the incidence rate ratio from the cancer registries for the same exposure contrast?

## Survey side (BRFSS 2022, file LLCP2022.ASC)
- Records: completed interviews (DISPCODE 1100) in the 22 states that carried the Cancer Survivorship module
  (FIPS 05 08 09 10 15 16 17 18 26 29 31 32 34 35 37 41 44 46 49 50 51 55; Puerto Rico excluded), excluding
  questionnaire version 20 in the six states where that version did not carry the module (08 09 26 31 34 49).
  Reason for completed interviews only: 99.9% of completed interviews reporting a cancer were asked the module,
  against 32% of partial interviews, and the healthy group has no module answer to show who was asked.
- Age: 20 to 79 years (_AGE80). The public file top-codes age at 80; registry age groups start at 20.
- Exposure: sex (_SEX), men against women, as in the manuscript's registry example.
- Disease: cancer other than non-melanoma skin cancer.
  - Healthy: CHCOCNC1 = 2 (never told "melanoma or any other types of cancer").
  - Case: CHCOCNC1 = 1. Answers 7, 9 and blank are excluded.
- Recent case, primary definition: a case with a valid age at diagnosis (CNCRAGE 1 to 97) equal to current age
  (told since the last birthday), unless the respondent reports one type of cancer and that type is
  non-melanoma skin (22) or skin of unknown kind (23). Cases with unknown age at diagnosis count as not recent.
  A case told a years ago is in this class with probability 1 - a (0 < a < 1): a window of under one year with
  triangular weight, effective length half a year.
- Recent case, secondary definitions:
  (S1) as primary and also exactly one reported type with the type known (the definition used in the other assessment);
  (S2) current age minus age at diagnosis of 0 or 1 (under two years; weight 1 in the first year, falling to 0 at two).
- Estimators: ROR = (recent_M/healthy_M)/(recent_F/healthy_F); POR = (cases_M/healthy_M)/(cases_F/healthy_F).
- Primary inference: weighted by _LLCPWT, 95% interval on the log scale from stratified linearisation
  (strata _STSTR, respondents as sampling units, with replacement, domain estimation with all records of a stratum kept).
- Secondary inference: unweighted counts with the manuscript's exact conditional interval (not design-based).

## Registry side (USCS 1999-2023 ASCII, BYAGE.TXT and BYAREA.TXT)
- Incidence, "All Cancer Sites Combined", All Races, United States, year 2022 (2021 and 2023 shown for stability).
- IDRR = (cases_M/population_M)/(cases_F/population_F) over the five-year groups from 20-24 to 75-79.
- Check on the choice of states: all-ages crude rate ratio in the 22 module states against the United States (BYAREA.TXT).

## Comparisons, all reported
1. Ages 20 to 79 (primary).
2. Ages 20 to 49.
3. Ages 50 to 79.
The registry sex ratio is expected to differ between the two age bands; the survey should reproduce that.

## How the result will be read
Descriptive. For each comparison: registry IDRR, POR with interval, ROR with interval, and whether the IDRR lies
inside the design-based interval of the ROR. With a few hundred recent cases the interval will be wide, so
containing the IDRR is weak evidence unless the POR lies clearly outside the same interval.
Known mismatches, to be stated with the result whatever it is: registry counts include second and later primaries
and use the whole population as denominator; the survey reaches only people alive, at home and able to answer;
age at diagnosis is self-reported in completed years; the registry figure is national, the survey covers 22 states.


## Post-analysis harmonization (16 November 2026)

The initial run omitted skin-only histories from the recent-onset numerator but counted them in the POR case numerator. Following independent review, skin-only histories (a single non-melanoma or unknown-type skin cancer) were removed from both eligible-cancer numerators and counted as having no eligible cancer in the denominator. This is a post-analysis definitional correction and **was not part of the original plan**. Updated point estimates, intervals, and expected outputs are in the final verification bundle.
