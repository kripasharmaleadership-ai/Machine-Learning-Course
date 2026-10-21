# Module 5 Assignment: A Model Comparison Report for Predictive Maintenance

> *"The best model is the one you can run, explain and trust, not just the one with the highest AUC."* A model comparison is a **report on four dimensions** (performance, explainability, speed and deployment complexity) that a non-specialist can follow and an engineer can verify.

## The scenario

A water utility runs **400 pumps in 4 plants**. Every Monday a model scores each pump's latest weekly telemetry, and the maintenance planner inspects the riskiest **10%** that week.

- An unplanned failure costs about $18,000; an inspection about $400.
- The model must run on the plant's **edge gateway**: limited memory, no internet.
- The planner must be able to **explain** the list to plant managers.
- The fleet grows, so the model must work on **pumps it has never seen**.

The data covers 16,000 weekly snapshots (400 pumps × 40 weeks). It is in `data/pump_telemetry.csv`, created on the first run of `assignment_starter.py`. **80 pumps are sealed as the test set.** Cross-validation is grouped by pump.

Reference solution for a *different* problem: [14_assignment_worked_example_model_report.py](14_assignment_worked_example_model_report.py), a telemarketing call list on real bank data. It shows five candidates, four dimensions, a leakage trap, drift and a written report. Study it first.

## The data

| column | notes |
|---|---|
| `machine_id`, `week` | pump identifier, week number |
| `plant`, `pump_type` | P1-P4; A, B, C |
| `install_age_years` | years since the pump was installed |
| `days_since_service` | days since the last service (reset by service or repair) |
| `temperature_c`, `vibration_mm_s`, `pressure_bar`, `rpm`, `load_factor` | the week's telemetry |
| `downtime_hours_next_7d` | hours of downtime recorded during the next 7 days |
| `failure_next_7d` | **TARGET**: the pump fails during the next 7 days (~4%) |

There are traps in this data. Finding them is part of the assignment.

## The requirements (fixed by the operations team)

| requirement | type | metric | threshold / weight |
|---|---|---|---|
| No future information | HARD gate | the leakage test | must hold |
| Works on new pumps | HARD gate | AUC, cross-validated by pump | ≥ 0.65 |
| Fits on the edge gateway | HARD gate | model size | ≤ 20 MB |
| Ranking quality | soft | grouped CV AUC | 20% |
| Failures caught in the top 10% | soft | recall when inspecting the top 10% | 20% |
| Explainability | soft | 1 = readable rules … 0.4 = forest + SHAP | 20% |
| Training time | soft | fit seconds | 5% |
| Latency | soft | ms per pump | 10% |
| Model size | soft | KB | 10% |
| Deployment complexity | soft | preprocessing steps + log10(learned numbers) + dependencies | 15% |

You may **add** requirements (and justify them). You may not loosen these.

## What to do

Work in **Part B** of [assignment_starter.py](assignment_starter.py). Part A (brief, data, requirements, harness and a baseline rule) and Part C (scorecard, rubric and report) are written for you.

1. **At least three candidates from at least two families**: a tree, a forest, a booster, logistic regression…
   - Each is a scikit-learn `Pipeline` on the raw DataFrame, registered with `evaluate(name, model, explainability, how_explained, preprocessing_steps, dependencies)`.
   - The harness measures all four dimensions: grouped CV, recall in the top 10%, fit time, latency, size, complexity and leakage.
2. **Leakage review:** for every column, ask *"is this known on Monday morning?"*
3. **Explain your recommended model** with `global_importance(name)` and `local_explanation(name, row, features)`. Group related columns into plain-language themes.
4. **Sensitivity analysis:** define at least two alternative weightings in `SCENARIOS`. Does the winner change?
5. **Open the test set once** (`open_test_set()`), after your choice is made on cross-validation.
6. **Write the report** in `REPORT`:
   - four sections (Performance, Explainability, Speed, Deployment Complexity), each ≥ 40 words, citing your numbers;
   - a Recommendation of ≥ 120 words that names the winner and states the trade-offs, the traps you found and what you would monitor.

## Deliverables

| # | File | Produced by |
|---|---|---|
| 1 | `outputs/assignment_pump_model_report.md` | starter (auto, when all checks pass) |
| 2 | `outputs/assignment_pump_model_report.json` | starter (auto, when all checks pass) |
| 3 | Your completed `assignment_starter.py` | you |
| 4 | **Stretch (optional):** an `XGBoostLite` candidate (Lab 05) and a discussion of what custom code does to deployment complexity | you |

## Grading rubric (100 points)

**Automated checks (40 pts).** These are printed by `assignment_starter.py`, Part C:

| check | pts |
|---|---|
| ≥ 3 candidates besides the baseline | 5 |
| ≥ 2 model families | 5 |
| No candidate uses future information | 6 |
| Requirements not loosened | 4 |
| Recommended model explained globally AND locally | 6 |
| ≥ 2 weight scenarios | 4 |
| Test set opened exactly once | 4 |
| Report: 4 sections ≥ 40 words, recommendation ≥ 120 words naming the winner with ≥ 3 numbers | 6 |

**Reviewer judgement (60 pts):**

| criterion | pts | what "excellent" looks like |
|---|---|---|
| Performance analysis | 12 | Grouped CV and test results are reported *and interpreted*: why grouping matters here, how noisy 80 test pumps are, what recall in the top 10% means for the planner |
| Explainability analysis | 12 | Global drivers in plain language ("vibration together with high temperature"); one pump's list placement explained; honest about post-hoc vs transparent models |
| Speed & deployment analysis | 12 | Size, latency and complexity compared and linked to the gateway; the price of a default random forest is noticed |
| Decision quality | 14 | The recommendation follows from gates and weights; trade-offs named; the sensitivity analysis is discussed; no re-choosing after the test set |
| Traps and limitations | 10 | The leaky column found and explained; the identity leak of random CV explained; new plants, sensor drift and the hidden pump-to-pump variation discussed |

## Hints

- **Why GroupKFold?** Each pump appears in 40 weeks. Random folds let a model recognise pumps it has already seen (Lab 06, 3.1). New pumps are what matter.
- **Size is a gate here.** A default random forest stores every leaf of every fully grown tree (Lab 04, 3.2). Leaf size, depth and the number of trees change the size enormously.
- **The failure mechanism is non-linear:** high vibration *and* high temperature together, overdue service, heavy load. Trees find interactions; a linear model needs them engineered (Lab 01, 3.1; Lab 08).
- **Explainability scores are judgement calls:** justify yours in `how_explained`. A depth-4 tree's 16 rules are more transparent than SHAP on 300 boosted trees.
- **The test set has only 80 pumps** (~130 failures). A model can look best on test after failing a gate on CV. That is noise; don't re-choose (Lab 07, 3.1).
- If a candidate scores AUC ≈ 1.0, stop and look for leakage before celebrating (Module 1, Lab 11).

## Stretch goals

- **Monotonic constraints** (Lab 05): more vibration or more days since service must never *lower* the risk. What does it cost in AUC, and what does it buy in trust?
- **Compile the tree** (Lab 13): export your best tree as if/else code. What happens to the deployment complexity score?
- **Escalation policy** (Lab 13): send pumps near the decision line to a human technician's judgement. How would you evaluate it?
