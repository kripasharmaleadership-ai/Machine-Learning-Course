# Module 4: Classification and Model Evaluation

[🏠 Course home](../README.md) · [📋 Syllabus](../SYLLABUS.md) · [← Module 3](../Module3_Regression_Mathematics_of_Learning) · **Module 4 of 8** · [Module 5 →](../Module5_Trees_Ensembles_Tuning_XAI)

Thirteen runnable labs and an assignment. Each lab builds from **EASY → MEDIUM → HARD**, and each HARD level ends with a "plot twist". The scenarios are loan defaults, card fraud, cancer biopsies, telecom churn, a customer-support AI assistant and sepsis screening in an emergency department.

The theme of the module: **a classifier gives you a score; the threshold turns it into a decision, and the decision has a price.** Accuracy rarely measures what the business cares about (Labs 03 and 05). The same model can lose to a one-line rule at the default threshold of 0.5 and beat it at a well-chosen one (Lab 11). And the classifiers inside LLM systems and agents are evaluated with exactly the same tools (Lab 12).

Every lab is a plain Python file with `# %%` cell markers:
- **Script mode:** `python 04_probabilities_thresholds_roc_pr.py` prints a narrated walkthrough and saves charts to `outputs/`.
- **Cell mode (best for recording):** open the file in VS Code and click **Run Cell** above each `# %%`. Charts appear inline.

## Setup

```bash
pip install -r requirements.txt
python run_all.py          # smoke-test all 13 labs + the assignment starter (~5 minutes)
```

Lab 11 downloads the real **IBM Telco Customer Churn** dataset from OpenML (data_id 42178) the first time it runs, then caches it in `data/telco_churn.csv`. Lab 08 uses the **Breast Cancer Wisconsin** and Lab 01 the **handwritten digits** datasets, both bundled with scikit-learn. Every other lab runs fully offline, and no API key is needed anywhere.

All randomness is seeded, so the numbers below come out the same on each run. Small differences can appear with other library versions (tested with scikit-learn 1.9, pandas 2.2, numpy 2.2). `imbalanced-learn` is **not** needed: SMOTE is written from scratch in the toolkit.

## Folder layout

| File | Purpose |
|---|---|
| `ml_lab_utils.py` | Console/plot helpers. The Telco loader. Synthetic datasets with **known true probabilities**: **loan applicants** (a known logistic formula), **card transactions** (1.6% fraud, a $ amount at risk, one hidden interaction), **support messages** (7 intents, held-out paraphrases, subtle attacks, label noise, out-of-scope questions), **ED visits** (sepsis, two hospitals, a leaky column) |
| `classification_toolkit.py` | A **reusable classification library** built up across the module, every function checked against scikit-learn: `binary_metrics()`, ROC / PR curves and AUC by hand, `threshold_sweep()`, `bayes_threshold()`, `threshold_strategies()` (7 strategies side by side), `calibration_table()` / ECE, `prior_shift()`, `smote()`, `LogisticRegressionGD` (from scratch), `bootstrap_compare()`, `segment_report()`, `ModelScorecard` (business gates + weights → decision memo) |
| `01` … `06` | Core concepts |
| `07` … `10` | The algorithms |
| `11_hands_on_lab_customer_churn.py` | The hands-on lab (real data, end to end) |
| `12_genai_connection_...py` | Classification inside LLM systems and agents |
| `13_assignment_worked_example_fraud_thresholds.py` | Reference solution: threshold strategies for card fraud |
| `ASSIGNMENT.md` + `assignment_starter.py` | The student assignment (sepsis screening) with an automated rubric |
| `run_all.py` | Runs every lab and reports pass/fail |
| `outputs/` | Charts, the saved churn model (threshold included), metrics JSON, model card, fraud decision memo |

> **Why synthetic data in most labs?** "Is this predicted probability *right*?" and "did logistic regression recover the true effect?" can only be *proven* when you know the true probability. Real data only ever shows you 0s and 1s. Lab 04 checks calibration against the truth; Lab 07 recovers the generating coefficients. Labs 01, 08 and 11 then use real data.

### Syllabus coverage

| Syllabus topic | Where |
|---|---|
| Binary and multiclass classification | 01 |
| Decision boundaries | 02 (and 07, 08, 09) |
| Probabilities and thresholds | 04, 06, 11, 13 |
| Class imbalance | 05 (and 11 Part 4) |
| False positives and false negatives | 03 (and 06, 11 Part 5.3) |
| Why accuracy can be misleading | 01, 03, 05 |
| Cost-sensitive model evaluation | 06 (and 11, 12, 13) |
| Logistic regression, KNN, decision tree, random forest, gradient boosting | 07, 08, 09, 10 |
| Confusion matrix, accuracy, precision, recall, specificity, F1 | 03 |
| ROC curve and ROC-AUC, precision-recall curve | 04 |
| Hands-on lab: churn: multiple classifiers, metrics, threshold tuning, imbalance, confusion matrix | 11 |
| GenAI: intent detection, prompt routing, query classification, safety filtering, tool selection, agent decisions | 12 |
| Assignment: threshold strategies (medical / fraud / churn) | 13 (fraud, solved) + ASSIGNMENT (medical) + stretch (churn) |

---

## Episode plan with the key on-screen moments

### Part A: Core concepts

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 01 | Binary and multiclass | `predict_proba` vs `predict`; a baseline | Softmax by hand; OvR vs OvO (45 models); multiclass confusion matrix; multi-label | The **closed world**: a classifier must always answer; the reject option; top-k accuracy | "Nobody defaults" is **85% accurate**. On random-noise images the digit model is **more** confident (0.99) than on real digits (0.89); the confidence filter rejects **4%** of the noise but **30%** of real digits |
| 02 | Decision boundaries | The logistic boundary as an equation; the threshold slides it | Five algorithms, five shapes (moons, circles) | Feature engineering bends a line; inductive bias (diagonal vs staircase); predictions **far from the data** | Circles: linear **0.50** → **0.99** with one r² feature. A diagonal boundary needs 32 tree leaves (96%); rotated 45° it needs 2. At (50, 50) the models say P = **0.00 to 1.00** |
| 03 | Confusion matrix and core metrics | 12 transactions by hand = sklearn; what FP / FN mean in 5 businesses | Four stakeholders, four metrics; why F1 is harmonic; F-beta | The **base-rate trap**; macro vs micro; F1 depends on the "positive" class; MCC | 97.7% accuracy with 52% recall. "Flag everything": arithmetic mean 0.51, F1 **0.03**. A 95%/95% test at 1% prevalence: precision **16%**. Fraud wave: recall fixed at 0.548, precision **36% → 83%** |
| 04 | Probabilities, thresholds, ROC, PR | Sigmoid; overlapping scores; the threshold sweep | ROC and PR curves by hand; AUC = P(positive ranked above negative); what AUC ignores | **Calibration** vs the true probabilities; ROC-AUC vs AP as positives get rarer; the region that matters | `p + 0.3` has the same AUC. Naive Bayes' top bin promises **96%** fraud, 11% occurs. Fraud rate 1.65% → 0.06%: AUC stays ~0.95, AP **0.42 → 0.05**. An AUC gap of 0.016 = **13% more frauds** caught at FPR 2% |
| 05 | Class imbalance, the accuracy paradox | 98.4% accurate, useless; stratified splits | Six strategies (weights, under/over-sampling, SMOTE from scratch, threshold moving) | **Oversampling before CV**; repairing probabilities (prior shift); few positives vs low ratio | All strategies: same AUC; weighting = a lower threshold (0.019), and it inflates the mean probability 0.016 → **0.175**. Oversample-then-CV: AP **1.00** vs real **0.23**. 25 → 1,600 frauds: AP 0.21 → 0.37; the ratio barely matters |
| 06 | Cost-sensitive evaluation | A cost matrix; total cost per threshold | Bayes threshold p\* derived = the searched one; it breaks with uncalibrated scores | **Per-transaction** thresholds; capacity (rank by expected $); regret when costs are wrong | No model $99.9k; best threshold **$54.4k**; choosing by F1 costs $5.3k more. Re-weighted model + p\*: **$204k**. Per-transaction p\*(amount) saves **$19.5k** more. Same 120 reviews, ranked by E[$]: **+$17.7k** stopped |

### Part B: The algorithms

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 07 | Logistic regression | Odds ratios; **recovers the true formula** | From scratch = sklearn; why not linear regression / MSE; C, L1 vs junk | **Perfect separation**; odds ratio ≠ risk change; class weights move only the intercept | Learned 0.537 vs true 0.55 per late payment (OR **1.71**). One late payment: **+3.8 points** of risk for a safe applicant, **+13.3** for a risky one. Balanced weights shift the intercept by **1.765** = log(weight ratio) **1.766** |
| 08 | K-nearest neighbours (real cancer biopsies) | KNN by hand = sklearn | k from 1 to 151; scaling; probabilities in steps of 1/k | Curse of dimensionality; latency grows with data; **KNN = retrieval** (vector search) | Unscaled 0.930 vs scaled 0.962. k = 151: malignant recall **0.63**. 300 junk features: missed cancers **6 → 18** (logistic 4 → 5) |
| 09 | Decision-tree classifier | A depth-2 tree as rules; Gini by hand | Depth, min leaf size, pruning (tuned on AP) | Trees find **interactions**; 0/1 probabilities; **instability** | Every leaf says "repay", even the 49% one. Night + foreign: actual **50%**, logistic **17%**, tree 38%. 526 of 630 flagged decisions change with a different 90% sample |
| 10 | Random forest + intro to gradient boosting | Bagging by hand | Forest knobs; **boosting from scratch** (Newton leaves) = sklearn; learning rate + early stopping | **The showdown**: 6 families, same folds, cost, speed, explainability | One tree AP 0.15 → 50 averaged **0.36**. Mean-residual booster AUC 0.88; with the Newton leaf step **0.944** = sklearn 0.943. Plot twist: **logistic regression beats boosting** (AP 0.419 vs 0.379); + one interaction feature 0.438, a statistical tie |

### Part C: Hands-on lab (`11_hands_on_lab_customer_churn.py`)

Real IBM Telco data (7,043 customers, 26.5% churn). A retention campaign: who do we call?

| Part | Level | What you do |
|---|---|---|
| 1 | EASY | Explore: month-to-month churns 15× more than two-year contracts; 11 blank TotalCharges explained; 22 identical rows kept, with a reason |
| 2 | EASY | Stratified split; baselines in **dollars**: call nobody (73% accurate, $0), call everybody ($222k), contract rule ($287k) |
| 3 | MEDIUM | **Five classifiers** in pipelines, out-of-fold. Logistic 0.846 ≈ forest 0.848 ≈ boosting 0.850 (fold sd 0.011) |
| 4 | MEDIUM | **Imbalance**: `class_weight` at 0.5 = the plain model at 0.26, but with broken probabilities |
| 5 | HARD | **Threshold tuning** for profit (7 strategies; p\* 0.175 ≈ profit optimum 0.173; `TunedThresholdClassifierCV` agrees); the **confusion matrix in business words**; segment audit; drivers; test once; model + threshold + model card saved |

Plot twist on the test set: the good model at the default 0.5 earns **$56k, less than the one-line contract rule ($67k)**. At the tuned threshold it earns **$71.6k**. The segment audit finds the blind spot: only **3%** of two-year-contract churners are ever called.

### Part D: Connection to GenAI (`12_genai_connection_intent_routing_safety_agents.py`)

| Level | GenAI component → classification problem | Key on-screen moment |
|---|---|---|
| EASY | **Intent detection** and **tool selection** as text classification; vs a keyword router | **95%** accurate on familiar phrasings, **53%** on new ones: the gap embeddings and LLMs close. Keywords 0.61, TF-IDF + logistic 0.79 |
| MEDIUM | **Safety filtering** (recall-first); **prompt routing** (small vs large LLM); **query classification** (out of scope) | 99.5% recall is unreachable: novel attacks score low (100% caught when familiar, 87% when new). Routing keeps quality 0.937 for **$3.0 vs $6.0** per 1,000 messages. A scope classifier trained on *imagined* off-topic questions catches **0%**; with 150 from the logs, **100%** |
| HARD | **Agent decision-making**: a cascade of classifiers + confidence-based escalation, priced in dollars | Act on everything $3,300, escalate everything $3,565, one global threshold $2,580, **per-tool risk thresholds $1,939** per 1,000 messages (refunds and cancellations demand 0.9+ confidence) |

### Part E: Assignment

- **`13_assignment_worked_example_fraud_thresholds.py`**: card-fraud operations (approve / review / auto-decline), with 20 analysts and a 2% customer-friction cap.
  - **Method:** requirements are written first: 3 hard gates and 4 weighted criteria. **Ten strategies** are tuned on validation: default, max F1, Youden, fixed recall, Bayes p\*, top-k capacity, per-amount p\*, two-tier, an expected-$ queue, and auto-decline + an expected-$ queue.
  - **Result:** the winner, **S9 (auto-decline + expected-$ queue)**, stops 69% of fraud dollars on test and beats the runner-up by **$4.2k** (CI $1.3k–$8.0k).
  - **Stress test:** in a **3× fraud wave nothing passes every gate** (S9 reaches 59.3% against the 60% target). A wave-mode fallback (auto-decline at p ≥ 0.4) restores 64%, and it goes into the memo.
- **`ASSIGNMENT.md` + `assignment_starter.py`**: students design threshold strategies for **sepsis screening** at two hospitals.
  - **Built-in trade-offs:** sensitivity ≥ 85%, ≤ 20 alerts per 100 visits, site parity, and a **leaky column** detected automatically. The baselines fail: qSOFA ≥ 2 has **26%** sensitivity and the model at 0.5 has 56%. Rural site B lacks lactate in 73% of visits.
  - **Uncertainty:** each policy prints a 95% lower bound on its sensitivity. A policy that just passes on validation can fail on the test month, and the rubric rewards saying so.
  - **Grading:** an automated rubric awards 40 points; reviewer judgement awards 60. The stretch goal applies the method to churn.

---

## Recording tips
- Run `python run_all.py` once before recording, so the charts, the Telco cache and all outputs exist.
- In cell mode, pause after each `>>` note. Those lines are written as talking points, with the numbers filled in from the run.
- Several labs show **honest negative results**:
  - Logistic regression beating boosting (Lab 10).
  - Ensembles tying with logistic regression on real data (Lab 11).
  - A safety filter that can't reach its recall target (Lab 12).
  - A fraud wave no policy survives (Lab 13).

  Lean into them: that's how real classification projects go.
- Lab 03's base-rate table and Lab 11's "the model at 0.5 loses to a one-line rule" are the two moments to slow down for.
- `run_all.py` doubles as a 5-minute pre-flight check after any library upgrade.
