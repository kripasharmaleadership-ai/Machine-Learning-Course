# Module 5: Trees, Ensembles, Tuning, and Explainable AI

[🏠 Course home](../README.md) · [📋 Syllabus](../SYLLABUS.md) · [← Module 4](../Module4_Classification_Model_Evaluation) · **Module 5 of 8** · [Module 6 →](../Module6_Unsupervised_Recommendations_Embeddings)

Fourteen runnable labs and an assignment. Each lab builds from **EASY → MEDIUM → HARD**, and each HARD level ends with a "plot twist". The scenarios are credit applicants, bike-rental demand, pump maintenance, the US census and a bank's telemarketing campaign.

The theme of the module: **build strong tree models, then earn the right to trust them.**
- **How they work:** trees decide by greedy questions (Lab 01); ensembles trade bias against variance (Labs 03–05).
- **How to improve them honestly:** cross-validation that can lie (Lab 06), the winner's curse of tuning (Lab 07), and error analysis that points at the DATA (Lab 09).
- **How to explain them responsibly:** importance (Lab 08), PDP/ICE/ALE (Lab 10) and SHAP (Lab 11), all checked against a **known truth**.
- **Trees and agents:** a decision tree is the blueprint of an agent workflow (Lab 13).

Every lab is a plain Python file with `# %%` cell markers:
- **Script mode:** `python 05_gradient_boosting_and_xgboost.py` prints a narrated walkthrough and saves charts to `outputs/`.
- **Cell mode (best for recording):** open the file in VS Code and click **Run Cell** above each `# %%`. Charts appear inline.

## Setup

```bash
pip install -r requirements.txt
python run_all.py          # smoke-test all 14 labs + the assignment starter (~6-7 minutes)
```

Two labs download real datasets from OpenML the first time they run, then cache them in `data/`:
- Lab 12: **Adult / Census Income** (data_id 1590).
- Lab 14: **Bank Marketing** (data_id 1461).

Every other lab runs fully offline. All randomness is seeded, so the numbers below come out the same on each run. Small differences can appear with other library versions (tested with scikit-learn 1.9, pandas 2.2, numpy 2.2).

**No XGBoost or SHAP installation needed.** The toolkit implements the XGBoost objective (`XGBoostLite`) and **exact** Shapley values from scratch, and both are checked against the libraries. If `xgboost` or `shap` is installed, Labs 05 and 11 add them to their comparisons automatically.

## Folder layout

| File | Purpose |
|---|---|
| `ml_lab_utils.py` | Console/plot helpers. Loaders for Adult and Bank Marketing. Synthetic data with **known truth**: **credit applicants** (a non-linear default probability, a proxy feature, a 200-level noise code), **bike rentals** (additive components: the ground truth for PDP and SHAP; a near-duplicate feature), **pump telemetry** (grouped by machine, a leaky column), plus a 14-row loan table for by-hand calculations |
| `tree_toolkit.py` | A **reusable library** built up across the module: entropy / Gini / information gain / gain ratio; `ScratchTree` (= sklearn's splits); `bias_variance_proba()`; `xgb_gain()` and `XGBoostLite` (second-order gains, λ, γ, subsampling, learned missing-value direction); permutation and drop-column importance; `pdp_ice()`; `shapley_exact()`; `slice_report()` (error analysis); `deployment_profile()`; `ModelScorecard` (requirements → report) |
| `01` … `11` | Core topics |
| `12_hands_on_lab_adult_income.py` | The hands-on lab (real data, end to end) |
| `13_agentic_connection_...py` | Decision trees vs agent workflows |
| `14_assignment_worked_example_model_report.py` | Reference solution: a model comparison report on real bank data |
| `ASSIGNMENT.md` + `assignment_starter.py` | The student assignment (predictive maintenance) with an automated rubric |
| `run_all.py` | Runs every lab and reports pass/fail |
| `outputs/` | Charts, the saved Adult model with metrics and model card, the bank comparison report (MD + JSON) |

### Syllabus coverage

| Syllabus topic | Where |
|---|---|
| How decision trees make decisions; entropy and information gain; Gini impurity | 01 |
| Tree depth and pruning; visualise a tree | 02 (and 12) |
| Bagging vs boosting; random forests; gradient boosting; XGBoost overview | 03, 04, 05 |
| Hyperparameters vs learned parameters; cross-validation | 06 |
| Grid search; random search | 07 |
| Feature importance | 08 (and 10-12) |
| Error analysis; data-centric model improvement | 09 (and 12) |
| Global vs local explanations; partial dependence | 10 |
| SHAP overview; responsible model interpretation | 11 (and 12) |
| Hands-on lab: tree, forest, boosting, CV, tuning, visualise, explain individual predictions, structured error analysis | 12 |
| Agentic AI: conditional branching, rules vs learned decisions, router agents, workflow orchestration, confidence-based escalation | 13 |
| Assignment: model comparison report (performance, explainability, speed, deployment complexity) | 14 (solved) + ASSIGNMENT |

---

## Episode plan with the key on-screen moments

### Part A: How trees work

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 01 | How trees decide | Entropy, information gain and Gini by hand for 14 loans | A 60-line CART tree **= scikit-learn exactly**; Gini vs entropy | **XOR** (no single question helps); information gain loves IDs; gain ratio | Root question 'employed' (0.396 bits). XOR: the tree is a coin flip (0.49) up to depth 6. A forest gets **0.94**, and one engineered feature 'A != B' gets **0.94** with ONE question. An application ID wins on information gain (0.985 bits) |
| 02 | Depth, pruning, visualisation | Depth sweep; read a tree and one decision path | Pre-pruning knobs; cost-complexity path + the **1-SE rule**; ranked business rules | Probabilities vs the TRUE probability; stability; the **price of interpretability** | The unlimited tree: 713 leaves, test AUC **0.61**, 100% of probabilities are 0 or 1. Pruning does NOT beat simple pre-pruning (0.776 vs 0.793). 8 rules capture **84%** of the achievable AUC gain; the forest 95% |

### Part B: Ensembles

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 03 | Bagging vs boosting | One deep tree vs 200 averaged; one stump vs 200 added | **bias² and variance MEASURED** over 20 training sets, 8 models | Bagging a stable learner; tree correlation formula; adding trees | Deep tree → bagging: variance **−90%**. Stump → boosting: bias² **−84%**. Bagged logistic regression = logistic regression, to 4 decimals. Boosting at lr 0.3 over-fits with more trees (Brier 0.098 → 0.122); a forest never does |
| 04 | Random forests | Three lines; OOB = CV = test | Each knob's effect; ExtraTrees; class weights | **OOB fooled by duplicates**; deployment cost; calibration vs truth | 63.2% of rows per tree (theory 1 − 1/e). Duplicated rows: OOB AUC **0.958** vs test 0.808. Default forest **44 MB**; leaf 30 + depth 12: **1.7 MB** with better AUC. Forest probabilities under-state the high-risk band |
| 05 | Gradient boosting & XGBoost | One XGBoost split by hand: g, h, leaf value −G/(H+λ), gain − γ | **XGBoostLite (ours) = sklearn** (AUC 0.840 / 0.840 / 0.841); λ, γ, min_child_weight, subsampling; early stopping | Informative **missing values**; **monotonic constraints** | Regularisation: 8,000 leaves → 748, best round 36 → 279. Missing income: a linear model needs the flag (+0.031 AUC); trees find the imputed constant themselves. Unconstrained: **13%** of applicants get LOWER risk with MORE debt; monotonic: 0% at no AUC cost |

### Part C: Model improvement

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 06 | Hyperparameters & cross-validation | Hyperparameters vs learned numbers, counted (a forest learns 119,201) | k-fold by hand; how noisy one split is; how many folds | **Three ways CV lies**: grouped rows, a regime change, preprocessing outside the folds | One 80/20 split: AUC 0.742-0.831 for the same model. Pumps: random CV **0.741**, GroupKFold 0.647, new pumps 0.598. Regime change: shuffled CV 0.776; forward folds collapse to **0.547**. Feature selection on 5,000 noise columns: CV AUC **0.825** on coin flips |
| 07 | Grid vs random search | GridSearchCV end to end | Grid vs random at equal budget (12 repeats); tuning a booster; successive halving | **Winner's curse**: inner CV vs nested CV vs reality; tuning on pure noise | Honest: the grid got lucky (its 100 sits next to the optimum, 113); random is insurance (worst within 0.006). Tuning adds +0.018 test AUC; halving: 3× the candidates in a third of the time. Search reports 0.779, nested CV 0.749, reality 0.729. Coin-flip labels, 200 rows: best setting **0.589** |
| 08 | Feature importance | Impurity importance of a forest | MDI vs permutation (train/test) vs drop-column, against the truth | Correlated twins; **proxies**; model dependence | MDI ranks pure noise **#5 and #6**; permutation ranks all three no-effect features last. Twins: each looks weak; shuffled together **0.114**. Without utilisation, 'open_cards' (no effect) ranks #5: "close 3 cards" moves the score, TRUE risk change **0** |
| 09 | Error analysis & data-centric | Slice the out-of-fold errors | A **region-blind** model exposes bad labels in R15; the thin young slice | **Model-centric vs data-centric scoreboard**; "cleaning" that hurts | R15 recorded 36% defaults vs 14% expected (2.7×). Tuning +0.002 AUC, a bigger model −0.012; neither fixes R15 over-prediction of **+16 points**. The label audit → −0.8 points; + young data: AUC 0.812 → 0.822. Deleting hard rows: worse on every metric |

### Part D: Explainable AI

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 10 | Global vs local, PDP / ICE | Importance + PDP by hand = sklearn = **the true curve?**; what-if for one hour | ICE reveals hour × workday; every feature's PDP vs truth | Correlated features: **impossible evaluations**; **ALE by hand**; the edge of the data | At 8:00 the PDP says 304 (workdays 350, weekends 203): no real day. 'feels_like_c' (no effect) gets a PDP range of 19. **48%** of temperature-PDP evaluations are physically impossible days. ALE beats PDP at the hot edge (−26 vs −8, truth −29) |
| 11 | SHAP & responsible interpretation | Shapley by hand: 8 coalitions, 6 orderings, = `shapley_exact` | **Exact SHAP vs the TRUE contributions** (correlation **0.99+**); waterfall; global mean \|SHAP\| | Twins split credit; the **Rashomon effect**; the background population | 26.25 + 18.75 + 5.00 = 70 − 20 exactly. Twins: temp +53.5, feels-like +8.1; grouped +60.7 (truth +63.7). Four models with AUC within 0.02 give one applicant **P = 0.14 to 0.61** and different reasons |

### Part E: Hands-on lab (`12_hands_on_lab_adult_income.py`)

Real Adult census data (48,842 people, income > $50k: 23.9%), for a financial-inclusion outreach programme.

| Part | Level | What you do |
|---|---|---|
| 1 | EASY | Explore; decide features BEFORE modelling: fnlwgt is a sampling weight, education duplicates education_num, sex and race are kept for auditing |
| 2 | EASY | Baseline; a depth-3 tree (CV AUC 0.863), printed and **visualised** (`outputs/12_adult_tree.png`) |
| 3 | MEDIUM | Tree, forest, boosting **tuned** with random search (inner CV), compared on fresh 5-fold **CV**: 0.909 / 0.917 / **0.927** |
| 4 | MEDIUM | **Explain**: permutation importance (capital gain, age, marital status...); **individual predictions** as exact-SHAP waterfalls and as a tree path |
| 5 | HARD | **Structured error analysis** (slices); **fairness audit by sex**; test opened once (boosting 0.930, beating the forest with CI +0.010 to +0.014); model + metrics + model card saved |

Plot twist: the model never sees "sex", yet recall for high-earning **women is 59% vs 66% for men**, because 'Husband' / 'Wife' carries sex into the model. Dropping the marital fields costs 0.037 AUC. The lab documents this as a policy decision, not a silent technical one.

### Part F: Connection to Agentic AI (`13_agentic_connection_trees_and_workflows.py`)

| Level | Tree concept → agent concept | Key on-screen moment |
|---|---|---|
| EASY | **Conditional branching**: a hand-written loan workflow IS a tree; its trace is the explanation | One trace per outcome: approve, request documents, decline, human review |
| MEDIUM | **Rules vs learned decisions**; a learned tree **compiled to if/else code**; a **router agent** | Learned decision **+$34 per application** vs the experts' rules; the generated code reproduces the model exactly; a router learned from noisy logs routes **99.9%** correctly (naive rule 36%) |
| HARD | **Workflow orchestration** (steps, budget, traces); **confidence-based escalation** from leaf size / forest disagreement | Escalation: $129 → **$159** per application. A human review is worth **+$143** per case near the decision line and **−$78** far from it |

### Part G: Assignment

- **`14_assignment_worked_example_model_report.py`**: a nightly call list on real Bank Marketing data, scored on **performance, explainability, speed and deployment complexity**.
  - **Traps:** the leaky `duration` model (CV AUC **0.953**) is rejected by the leakage test. Strong drift: 7% → 32% subscriptions.
  - **Plot twist:** **logistic regression wins the report under every weighting**. It is not the CV leader, but it holds up better on recent months (+0.020 vs boosting, CI +0.011 to +0.030), weighs 3 KB instead of 38 MB, and is readable.
  - **Output:** the report (`outputs/14_bank_model_comparison_report.md`) passes 9/9 rubric checks.
- **`ASSIGNMENT.md` + `assignment_starter.py`**: students write the same report for **predictive maintenance**: 400 pumps, an inspection list of the top 10%, and an edge gateway.
  - **Built-in trade-offs:** grouped CV by pump and 80 unseen test pumps; a leaky downtime column (AUC ≈ 1.0); a **20 MB size gate** that rejects a default random forest (33 MB); a depth-4 tree that just misses the AUC gate on CV but looks best on the small test set.
  - **Grading:** an automated rubric awards 40 points; reviewer judgement awards 60.

---

## Recording tips
- Run `python run_all.py` once before recording, so the charts, the dataset caches and all outputs exist.
- In cell mode, pause after each `>>` note. Those lines are written as talking points, with the numbers filled in from the run.
- The synthetic datasets have a **known truth**. Lean on it: "we can check this explanation against reality" is the thread from Lab 02 to Lab 11.
- Several labs show **honest negative results**:
  - Pruning not beating pre-pruning (Lab 02).
  - A lucky grid (Lab 07).
  - Isotonic calibration not fixing the top band (Lab 04).
  - Tuning that can't fix bad labels (Lab 09).
  - The simplest model winning the report (Lab 14).

  That's how real projects go.
- Lab 01's XOR table, Lab 06's collapsing forward fold, Lab 09's scoreboard and Lab 11's Rashomon table are the moments to slow down for.
