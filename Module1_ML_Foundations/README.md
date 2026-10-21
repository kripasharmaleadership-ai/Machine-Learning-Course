# Module 1: Machine Learning Foundations and Problem Formulation

Fourteen runnable labs. Each one builds from **EASY → MEDIUM → HARD** and uses business scenarios (telecom churn, fraud, e-commerce, retention offers), not toy examples.

Every lab is a plain Python file with `# %%` cell markers:
- **Script mode:** `python 07_train_validation_test.py` prints a narrated walkthrough and saves charts to `outputs/`.
- **Cell mode (best for recording):** open the file in VS Code and click **Run Cell** above each `# %%`. Charts appear inline.

## Setup

```bash
pip install -r requirements.txt
python run_all.py          # smoke-test all 14 labs (~3 minutes)
```

Lab 14 downloads the real Titanic dataset from OpenML the first time it runs, then caches it in `data/titanic.csv`. Every other lab runs fully offline. All randomness is seeded (`RANDOM_STATE = 42`), so the numbers below come out the same on each run. Small differences can appear with other library versions.

## Folder layout

| File | Purpose |
|---|---|
| `ml_lab_utils.py` | Shared helpers: console formatting, plot saving, Titanic loader, synthetic telco/e-commerce data with **known ground truth** |
| `01` … `08` | Core topics |
| `09` … `13` | Important practical topics |
| `14_hands_on_lab_titanic.py` | The hands-on lab (real dataset, end to end) |
| `run_all.py` | Runs every lab and reports pass/fail |
| `outputs/` | Charts, saved models, model registry, model cards, metrics JSON, pre-mortem report |

> **Why synthetic data in Labs 03–13?** Leakage, drift and causation can only be *proven* when you know the true answer. The generators in `ml_lab_utils.py` contain the "laws of nature" for each scenario, so every claim can be checked against ground truth. Lab 14 then applies everything to real data.

---

## Episode plan with the key on-screen moments

### Part A: Core topics

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 01 | What is ML? | Pizza delivery: a manager's rule vs a learned rule | Gradient descent from scratch (what `.fit()` does) | Mitchell's E/T/P definition shown as a learning curve; memorization vs generalization | The model discovers `6.98 + 2.87·km` (true rule: `8 + 2.7·km`); 1-NN scores 100% on training data but only 92% on test data |
| 02 | Traditional vs ML vs GenAI | Keyword spam rules vs Naive Bayes | Spammers switch to crypto: retraining vs rewriting code | A language model built from scratch, temperature, hallucination; decision guide | Rules get 62.5% vs ML 100%; the model generates *"your ticket has been shipped today"* |
| 03 | AI / ML / DL / GenAI / Agentic | Expert-system loan rules vs logistic regression | Neural net on circles; a GenAI model that draws new digits | Retention **agent** with tools, budget, human approval and audit log | AUC 0.67 → 0.81; linear 49% vs NN 98%; agent v1 $642 → v2 **$903** with the same budget |
| 04 | Types of ML | Supervised vs unsupervised on iris | Semi-supervised (5% labels), anomaly detection, self-supervised | Bandits for email subject lines; Q-learning warehouse robot | Self-training 82% → 89%; Thompson sampling regret 22 vs random 449; robot finds the optimal 7-step path |
| 05 | Features, labels, observations, models, predictions | Annotated house-price table | Column roles, feature engineering, `predict` vs `predict_proba`, model as artifact | **Feature contract** vs training/serving skew | A CRM spelling change plus price in cents turns P(churn) from **0.07 into 1.00** silently; the contract rejects it |
| 06 | Regression / classification / clustering / recommendation | One mini-example of each | One e-commerce business framed 4 ways, with correct metrics and baselines | Recommender: popularity vs item-kNN vs matrix factorization, HitRate@10 | Silhouette picks k=3 (ARI 0.96); item-kNN 0.33 vs popularity 0.18 vs random 0.035 |
| 07 | Train / validation / test | Overfitting; why stratify | Tune on validation, K-fold CV, **test-set peeking** simulation | **Group** leakage and **time** splits with walk-forward validation | The overfit tree does *worse than guessing*; peeking finds "signal" in noise; AUC **0.39** right after a regime change |
| 08 | End-to-end lifecycle | The whole loop in ~40 lines | Data validation, CV model comparison, profit-based threshold | Registry plus model card, serving, **PSI drift monitoring**, champion/challenger | Threshold 0.10 roughly doubles profit vs 0.5; price PSI = 1.43 fires the retrain trigger |

### Part B: Important practical topics

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 09 | Business problem → ML problem | "Does this need ML?" screening | Problem framing canvas (churn, fraud, stock-outs) | Point-in-time labels from event logs; **uplift vs churn** framing | Leaky pipeline: 0.999 offline → **0.348** in production; uplift saves 718 customers vs 202 for churn-risk targeting (worse than random) |
| 10 | Inputs, outputs, constraints, metrics | ML spec plus feature availability check | Accuracy paradox on fraud; confusion matrix in dollars | Model selection under capacity, latency SLA, fairness and profit constraints | Dummy is 99.4% "accurate" with 0 recall; ROC-AUC 0.90 vs PR-AUC 0.19; profit vs call-list size |
| 11 | Data leakage | Target leakage (fake AUC 1.00) | Feature selection and target encoding on pure noise | Automated **leakage audit** tool | Leaky model AUC 1.00 → 0.62 on Monday's data; 77% accuracy on random noise; 9.9% duplicate rows caught |
| 12 | Correlation vs causation | Ice cream and drownings; spurious trends | Retention calls "increase churn" (Simpson's paradox) | Naive vs regression vs IPW vs A/B test against the true effect; predictive ≠ lever | Naive +5.4 pts vs true **−5.8 pts**; dropping one confounder halves the estimate; "ban tickets" promises 17% churn, reality 28% |
| 13 | Why ML projects fail | Project health scorecard | No baseline, noisy and biased labels, broken join | 12 months of silent decay, then an auto-generated **pre-mortem** | Team's dashboard says AUC 0.87, reality is 0.70; monitoring is worth **$77k/year** |

### Part C: Hands-on lab (`14_hands_on_lab_titanic.py`)

| Part | Level | What you do |
|---|---|---|
| 1 | EASY | Explore with pandas: shape, dtypes, `describe`, missing values, target balance, cardinality |
| 2 | EASY | Auto-profile column types, then a human data dictionary: numerical / categorical / text / ID / **target** / **leaky** (`boat`, `body`) |
| 3 | MEDIUM | EDA: survival by sex and class, interactions, age/fare patterns, informative missingness, feature ideas (title, family size, deck) plus a 4-panel chart |
| 4 | MEDIUM | Stratified train/test split with a verification table; seal the test set |
| 5 | HARD | Baseline ladder (majority → rules → logistic regression → random forest) with a full `Pipeline`, leakage trap, final test, coefficients, per-segment error analysis, saved model plus metrics JSON, predictions for Jack and Rose |

Key results: the "women survive" rule gets 80.2% test accuracy, logistic regression 84.0% (ROC-AUC 0.88), and keeping `boat`/`body` gives a fake **97.8%**.

---

## Recording tips
- Run each lab once before recording so charts and cached data exist.
- In cell mode, pause after each `>>` note. Those lines are written as talking points.
- Each HARD level has one "plot twist" result (the column above). Build the video towards it.
- `run_all.py` is a 3-minute pre-flight check after any library upgrade.
