"""
LAB 10 - Defining Inputs, Outputs, Constraints and Success Metrics
==================================================================

THE ML SPEC (write it BEFORE building anything)

    INPUTS       : which features, from which system, available WHEN, with what freshness
    OUTPUT       : a class? a probability? a ranked list? a number with an interval?
    CONSTRAINTS  : latency, throughput, cost, explainability, fairness, privacy, capacity
    METRICS      : a HIERARCHY, not one number
                     North-star business metric   (revenue retained, fraud $ prevented)
                       <- online/proxy metric     (precision of the call list, approval rate)
                         <- offline model metric  (PR-AUC, recall@k, MAE)
                     + GUARDRAIL metrics that must NOT get worse (complaints, fairness gap, latency)

LEVELS
    EASY   : Write the spec for churn; check which inputs are really available at prediction time.
    MEDIUM : Fraud (0.5% positives): why accuracy lies, and what to use instead.
    HARD   : Pick a model under REAL constraints - profit-optimal threshold, team capacity
             (precision@K), latency SLA, and a fairness guardrail - in one selection matrix.
"""
# %%
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, TELCO_CATEGORICAL, TELCO_NUMERIC, TELCO_TARGET,
                          banner, make_telco_churn, note, plt, save_fig, show, step)

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): the ML spec for churn
# ============================================================================
banner("Write the ML spec before the model", "EASY")
ml_spec = {
    "use_case": "Weekly churn call list",
    "inputs": {
        "source_systems": ["CRM (contract, tenure)", "Billing (charges)", "Support desk (tickets)"],
        "freshness": "snapshot every Monday 06:00",
        "excluded": ["cancellation_reason", "days_since_last_payment", "retention_offer_made"],
    },
    "output": {"type": "probability of churn in 90 days", "consumer": "retention team call list (top-K)"},
    "constraints": {
        "capacity": "300 calls per week",
        "latency": "batch job < 30 min; real-time lookups < 50 ms p99",
        "explainability": "top 3 reasons per customer for the call agent",
        "fairness": "recall gap between senior / non-senior customers < 10 pts",
        "privacy": "no protected attributes used as features except where legally reviewed",
    },
    "metrics": {
        "north_star": "revenue retained (vs randomized holdout group)",
        "proxy": "precision@300 (share of called customers who were really at risk)",
        "offline": ["PR-AUC", "recall@300"],
        "guardrails": ["complaint rate", "fairness recall gap", "p99 latency"],
    },
}
print(json.dumps(ml_spec, indent=2))

step("Input availability check: is each feature REALLY known at prediction time?")
availability = pd.DataFrame([
    ["tenure_months", "CRM", "yes", "daily", "OK"],
    ["monthly_charges", "Billing", "yes", "monthly", "OK"],
    ["support_tickets_90d", "Support desk", "yes", "hourly", "OK"],
    ["total_charges", "Billing", "yes (2% missing)", "monthly", "OK - impute"],
    ["days_since_last_payment", "Billing", "only AFTER the event", "-", "REJECT (leakage)"],
    ["cancellation_reason", "Cancel form", "only AFTER the event", "-", "REJECT (leakage)"],
    ["nps_survey_score", "Survey vendor", "yes, 6-week delay", "quarterly", "REJECT (too stale)"],
], columns=["feature", "source", "known at prediction time?", "refresh", "decision"])
print(availability.to_string(index=False))
note("""A feature is only valid if it exists, with the same definition and
freshness, at the moment the model is called. Many projects die when the
'great feature' from the data warehouse turns out to be unavailable online.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): metrics on imbalanced data - why accuracy lies
# ============================================================================
banner("Fraud detection: ~0.6% positives - the accuracy paradox", "MEDIUM")
n = 40_000
amount = rng.lognormal(3.5, 1.0, n)
night = rng.binomial(1, 0.15, n)
foreign = rng.binomial(1, 0.08, n)
new_device = rng.binomial(1, 0.1, n)
logit = -10 + 0.7 * np.log(amount) + 2.2 * night + 2.6 * foreign + 2.8 * new_device
fraud = rng.binomial(1, 1 / (1 + np.exp(-logit)))
F = pd.DataFrame({"log_amount": np.log(amount), "night": night, "foreign": foreign, "new_device": new_device})
show("Transactions / fraud rate", f"{n:,} / {fraud.mean():.2%}")
F_tr, F_te, f_tr, f_te = train_test_split(F, fraud, test_size=0.3, stratify=fraud, random_state=RANDOM_STATE)

always_ok = DummyClassifier(strategy="most_frequent").fit(F_tr, f_tr)
fraud_model = LogisticRegression(max_iter=1000).fit(F_tr, f_tr)
p_fraud = fraud_model.predict_proba(F_te)[:, 1]

rows = []
for name, pred, score in [
    ("'Never fraud' dummy", always_ok.predict(F_te), always_ok.predict_proba(F_te)[:, 1]),
    ("Model @ default 0.5", (p_fraud >= 0.5).astype(int), p_fraud),
    ("Model @ 0.05", (p_fraud >= 0.05).astype(int), p_fraud),
]:
    rows.append({"approach": name, "accuracy": accuracy_score(f_te, pred),
                 "precision": precision_score(f_te, pred, zero_division=0),
                 "recall": recall_score(f_te, pred), "F1": f1_score(f_te, pred),
                 "ROC-AUC": roc_auc_score(f_te, score), "PR-AUC": average_precision_score(f_te, score)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
note(f"""The dummy is {1 - f_te.mean():.1%} 'accurate' and catches ZERO fraud. With rare events use:
  PRECISION : of the transactions we block, how many were fraud?
  RECALL    : of all fraud, how much did we catch?
  PR-AUC    : precision/recall quality across ALL thresholds (baseline = fraud rate)
ROC-AUC {roc_auc_score(f_te, p_fraud):.2f} looks excellent; PR-AUC {average_precision_score(f_te, p_fraud):.2f} tells the truth: most flagged
transactions are still legitimate. On extreme imbalance, report PR-AUC.""", keep_lines=True)

step("Confusion matrix in DOLLARS (what the CFO cares about)")
tn, fp, fn, tp = confusion_matrix(f_te, (p_fraud >= 0.05).astype(int)).ravel()
AVG_FRAUD_LOSS, FRICTION_COST = 450, 8          # $ lost per missed fraud, $ cost of blocking a good customer
print(f"  caught fraud (TP)      {tp:>6}  -> saved  ${tp * AVG_FRAUD_LOSS:>10,}")
print(f"  missed fraud (FN)      {fn:>6}  -> lost   ${fn * AVG_FRAUD_LOSS:>10,}")
print(f"  blocked good (FP)      {fp:>6}  -> cost   ${fp * FRICTION_COST:>10,}")
print(f"  approved good (TN)     {tn:>6}  -> $0")
note("""FN and FP have very different prices. The right metric is the one that
reflects THOSE prices - which only the business can tell you.""")

# %% =========================================================================
# LEVEL 3 (HARD): model selection under real constraints
# ============================================================================
banner("Choosing a model under capacity, latency, fairness and profit constraints", "HARD")
df = make_telco_churn(n=12_000)
FEATURES = TELCO_NUMERIC + TELCO_CATEGORICAL
X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TELCO_TARGET], test_size=0.3,
                                          stratify=df[TELCO_TARGET], random_state=RANDOM_STATE)
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), TELCO_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), TELCO_CATEGORICAL),
])
candidates = {
    "logistic_regression": LogisticRegression(max_iter=2000),
    "random_forest_800": RandomForestClassifier(n_estimators=800, min_samples_leaf=5,
                                                random_state=RANDOM_STATE, n_jobs=1),
    "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                             max_iter=300, random_state=RANDOM_STATE),
}
CAPACITY_K = 300                 # retention team calls per week (on this test cohort)
LATENCY_SLA_MS = 50              # p99 for a single real-time prediction
FAIRNESS_MAX_GAP = 0.10          # max recall gap senior vs non-senior
OFFER_COST, SAVED_VALUE, SAVE_RATE = 25, 700, 0.3


def p99_latency_ms(model, X, n_calls: int = 150) -> float:
    rows_ = [X.iloc[[i]] for i in range(n_calls)]
    times = []
    for r in rows_:
        t0 = time.perf_counter()
        model.predict_proba(r)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 99))


selection = []
for name, clf in candidates.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    top_k = np.argsort(-p)[:CAPACITY_K]                     # who the team will actually call
    called = np.zeros(len(p), dtype=bool)
    called[top_k] = True
    y = y_te.to_numpy()
    senior = X_te.senior_citizen.to_numpy() == 1
    recall_senior = (called & (y == 1) & senior).sum() / max((y[senior] == 1).sum(), 1)
    recall_other = (called & (y == 1) & ~senior).sum() / max((y[~senior] == 1).sum(), 1)
    tp = (called & (y == 1)).sum()
    selection.append({
        "model": name,
        "PR-AUC": average_precision_score(y, p),
        f"precision@{CAPACITY_K}": tp / CAPACITY_K,
        "profit_$": tp * SAVE_RATE * SAVED_VALUE - CAPACITY_K * OFFER_COST,
        "p99_latency_ms": p99_latency_ms(pipe, X_te),
        "fair_recall_gap": abs(recall_senior - recall_other),
    })
sel = pd.DataFrame(selection).set_index("model")
sel["meets_latency"] = sel.p99_latency_ms <= LATENCY_SLA_MS
sel["meets_fairness"] = sel.fair_recall_gap <= FAIRNESS_MAX_GAP
sel["eligible"] = sel.meets_latency & sel.meets_fairness
print(sel.round(3).to_string())
# Among eligible models: anything within 5% of the best profit is a statistical tie
# (on 300 calls, 5% is ~10 customers - well inside run-to-run noise),
# and ties go to the SIMPLEST model (cheaper to run, easier to explain).
SIMPLICITY_ORDER = ["logistic_regression", "hist_gradient_boosting", "random_forest_800"]
eligible = sel[sel.eligible]
near_best = eligible[eligible["profit_$"] >= 0.95 * eligible["profit_$"].max()]
winner = min(near_best.index, key=SIMPLICITY_ORDER.index)
show("Models within 5% of best profit", list(near_best.index))
show("Selected model (simplest of those)", winner)
note(f"""How to read the selection matrix:
  1. Constraints are PASS/FAIL gates (latency <= {LATENCY_SLA_MS} ms, fairness gap <= {FAIRNESS_MAX_GAP}).
  2. Among models that pass, choose by the BUSINESS metric (profit at the team's capacity);
     near-ties go to the simplest model.
  3. Tiny PR-AUC differences rarely matter; a model that breaks the SLA never ships.
  4. Fairness here: senior customers churn more, so a risk ranking may call them
     disproportionately often - check the gap and discuss it with the business
     and legal teams; do not just 'optimize it away'.""", keep_lines=True)

step("Why capacity changes the right threshold: precision and recall vs list size")
best_pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]).fit(X_tr, y_tr)
p = best_pipe.predict_proba(X_te)[:, 1]
order = np.argsort(-p)
y = y_te.to_numpy()
ks = [100, 300, 600, 1000, 1500, 2000]
for k in ks:
    tp = y[order[:k]].sum()
    print(f"  call top {k:>4}: precision={tp / k:.2f}  recall={tp / y.sum():.2f}  "
          f"profit=${tp * SAVE_RATE * SAVED_VALUE - k * OFFER_COST:>8,.0f}  (threshold ~{p[order[k - 1]]:.2f})")
note("""With unlimited capacity the profit-optimal list is much longer than 300.
That is a business insight the model produced: 'hiring 2 more retention agents
pays for itself'. Good metric design turns a model into a decision tool.""")

if HAS_MPL:
    ks_full = np.arange(50, len(p), 50)
    profits = [y[order[:k]].sum() * SAVE_RATE * SAVED_VALUE - k * OFFER_COST for k in ks_full]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ks_full, profits)
    ax.axvline(CAPACITY_K, ls="--", c="grey", label=f"team capacity = {CAPACITY_K}")
    ax.set(xlabel="customers called (top-K by risk)", ylabel="expected profit ($)",
           title="Profit vs call-list size")
    ax.legend()
    save_fig(fig, "10_hard_profit_vs_k")

banner("KEY TAKEAWAYS")
print("""  1. Write the spec first: inputs (available WHEN?), output type, constraints, metrics.
  2. Metrics are a hierarchy: business north-star <- proxy <- offline, plus guardrails.
  3. On imbalanced data, accuracy lies - use precision, recall, PR-AUC and dollar costs.
  4. Constraints (latency, capacity, fairness) are gates; business value picks the winner.""")
