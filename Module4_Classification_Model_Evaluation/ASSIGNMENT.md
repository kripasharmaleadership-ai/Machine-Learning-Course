# Module 4 Assignment: Design Threshold Strategies for Sepsis Screening

> *"The model gives you a probability. The threshold strategy decides who gets treated."* In a hospital, choosing that threshold is a clinical-safety decision. It has to be designed, compared and defended with numbers, not left at the default 0.5.

## The scenario

You've joined the clinical data-science team of a regional health system. Its two emergency departments want an **early-warning alert for sepsis** at triage. A risk model already exists (logistic regression on triage vitals and labs, validation ROC-AUC ≈ 0.94). The clinical safety committee asks you to **design and compare threshold strategies** that decide which patients trigger an alert, and to recommend one.

- **Site A** is a city hospital.
- **Site B** is rural. Point-of-care lactate is usually not available there (73% missing vs 14% at site A).

The data covers 14,000 adult ED visits over 180 days (`data/ed_visits.csv`, created on the first run of `assignment_starter.py`):
- **Training:** days 0–119, used for the model.
- **Validation:** days 120–149. Tune every threshold here.
- **Test:** the last 30 days, sealed. Open it **once**.

Reference solution for a *different* scenario: [13_assignment_worked_example_fraud_thresholds.py](13_assignment_worked_example_fraud_thresholds.py) (card fraud: ten strategies, a two-tier triage, stress tests, a memo). Study it first.

## The data

| column | notes |
|---|---|
| `visit_id`, `arrival_time` | identifier, timestamp |
| `site` | A (city) or B (rural) |
| `age`, `comorbidities`, `nursing_home`, `arrived_by_ambulance` | patient context |
| `heart_rate`, `resp_rate`, `temp_c`, `systolic_bp`, `spo2` | triage vitals |
| `wbc`, `lactate` | labs (`lactate` often missing at site B) |
| `altered_mental_status` | 1 if confused or drowsy at triage |
| `antibiotics_within_1h` | 1 if antibiotics were given within an hour of arrival |
| `sepsis` | **TARGET**: sepsis confirmed within 48 h (~7% of visits) |

There are traps in this data. Finding them is part of the assignment.

## The requirements (fixed by the clinical safety committee)

| requirement | type | metric | threshold / weight |
|---|---|---|---|
| Sensitivity | HARD gate | share of septic patients alerted | ≥ 85% |
| Alert budget | HARD gate | alerts per 100 visits | ≤ 20 |
| Site parity | HARD gate | sensitivity at site B ÷ sensitivity at site A | ≥ 0.90 |
| Triage-time information only | HARD gate | the policy must not use anything known only after triage | must hold |
| Missed sepsis | soft | misses per 1,000 visits | 40% |
| Alert load | soft | alerts per 100 visits | 30% |
| Positive predictive value | soft | share of alerts that are sepsis | 15% |
| Simplicity | soft | 1 = one threshold, 0.7 = a combination, 0.4 = per-segment rules | 15% |

The committee also estimates that **one missed sepsis is as bad as 30 false alerts** (`COST_RATIO = 30`). You may **add** requirements (and justify them). You may not loosen these.

## What to do

Work in **Part B** of [assignment_starter.py](assignment_starter.py). Part A (data, requirements, risk model, the qSOFA baseline, evaluation harness) and Part C (scorecard, rubric, memo) are written for you.

1. **At least four threshold strategies of at least three kinds.** Register each one with `evaluate_policy(name, decide, kind, simplicity, why)`.
   - `decide(d)` receives the validation visits plus a `risk` column and returns a boolean alert per visit.
   - `kind` is one of `statistical`, `fixed-sensitivity`, `alert-budget`, `cost-based`, `segment-aware`, `combination`, `other`.
   - Every number inside a strategy must come from **training or validation** data.
2. **Leakage review:** for every column, ask *"is this known at triage?"* The harness shuffles the suspicious column and checks whether your alerts change.
3. **Site audit:** `site_report(name)` shows sensitivity, PPV and alert rate per hospital. Does your policy fail rural patients? Why, and what did you change?
4. **Uncertainty:** each policy prints a 95% lower bound for its sensitivity (validation has only ~160 septic patients). Does your recommendation pass the gate with a margin, or only on paper?
5. **Sensitivity analysis:** define at least two alternative weightings in `SCENARIOS`. Does the winner change?
6. **Open the test set once** with `open_test_set()`, *after* your choice is made on validation. Report honestly whether the winner still passes every gate on the new month.
7. **Justify:** write `JUSTIFICATION` (≥ 150 words). It should:
   - name the winner and cite validation and test numbers;
   - say which hard gates decided;
   - report what the site audit and the sensitivity analysis showed;
   - state the limitations: flu season doubles the prevalence; a third hospital has no lactate testing at all.

## Deliverables

| # | File | Produced by |
|---|---|---|
| 1 | `outputs/assignment_sepsis_policy_memo.md` | starter (auto, when all checks pass) |
| 2 | `outputs/assignment_sepsis_policy.json` | starter (auto, when all checks pass) |
| 3 | Your completed `assignment_starter.py` | you |
| 4 | **Stretch (optional):** a strategy for a second scenario, **customer churn** (reuse Lab 11's model and economics), compared with the same method | you |

## Grading rubric (100 points)

**Automated checks (40 pts).** These are printed by `assignment_starter.py`, Part C:

| check | pts |
|---|---|
| ≥ 4 strategies besides the baselines | 6 |
| ≥ 3 different kinds of strategy | 6 |
| No strategy uses post-triage information | 6 |
| Requirements not loosened | 4 |
| Site audit done for the recommended policy | 4 |
| ≥ 2 weight scenarios in the sensitivity analysis | 4 |
| Test set opened exactly once | 4 |
| Justification: ≥ 150 words, names the winner, cites ≥ 3 numbers | 6 |

**Reviewer judgement (60 pts):**

| criterion | pts | what "excellent" looks like |
|---|---|---|
| Range of strategies | 15 | The strategies differ in *how* they think: a sensitivity target, an alert budget, a cost-based p\* = 1 / (1 + 30), a site-aware rule, a model + clinical-rule combination. Each comes with a one-line rationale |
| Clinical and business reasoning | 15 | The choice follows from the gates and weights; trade-offs are named in clinical terms ("S3 catches 4 more septic patients per 1,000 visits but adds 6 alerts per 100: beyond the budget") |
| Diagnosis | 10 | The leaky column is found and excluded with a reason; the site B gap is explained (missing lactate) and addressed, or its residual risk is stated |
| Honesty about uncertainty | 15 | Margins against the gates are reported with the CI lower bound; if the winner fails a gate on the test month, the memo says so and proposes a fix (more margin, re-tuning, monitoring) instead of hiding it |
| Limitations and monitoring | 5 | Prevalence shift (flu season), new sites, drift; what to monitor weekly and when to re-tune |

## Hints

- Start from the score distribution: `with_risk(val)` gives you every validation visit with its `risk`.
- **Fixed sensitivity:** the threshold that catches 90% of the validation cases is the 10th percentile of `risk` among septic patients.
- **Alert budget:** the threshold that alerts on 18% of visits is the 82nd percentile of `risk` over all visits.
- **Cost-based:** with calibrated probabilities, alert when `risk > 1 / (1 + COST_RATIO)` (Lab 06). Is this model calibrated? (Lab 04)
- **Site-aware:** a lower threshold at site B can restore parity, but it costs alerts. Is a different threshold per site *fair*? Discuss it.
- **Combination:** `(risk >= t) | (qsofa(d) >= 2)` adds the clinicians' own rule as a safety net.
- A policy that sits exactly on a gate on validation is a coin flip on the next month (Lab 03: precision and recall move with the sample). Leave a margin.
- If one strategy looks *too good to be true*, check which columns it uses (Module 1, Lab 11).

## Stretch goals

- **Recalibrate for flu season:** prevalence doubles. Use `prior_shift()` (Lab 05) to adjust the probabilities. What happens to each strategy's alert load?
- **Two-tier policy:** "alert" (physician now) vs "re-check vitals in 1 hour" (nurse) for a middle band of risk. How would you evaluate a three-action policy? (See S7 / S9 in the worked example.)
- **Different model, same policy:** replace `MODEL` with gradient boosting (Lab 10). Does the best *strategy* change, or only its thresholds?
