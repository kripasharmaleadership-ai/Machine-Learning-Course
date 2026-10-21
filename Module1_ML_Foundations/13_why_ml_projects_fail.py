"""
LAB 13 - Common Reasons ML Projects Fail
========================================

Industry surveys repeatedly report that most ML projects never reach production
or never deliver value. The causes are rarely "the algorithm was not fancy enough".

THE USUAL SUSPECTS
     1. Unclear business problem / no decision attached to the prediction
     2. No baseline - nobody knows if the model beats a simple rule
     3. Wrong metric (accuracy on imbalanced data, offline metric unrelated to money)
     4. Data leakage -> great offline, poor online
     5. Poor label quality (noisy, inconsistent, delayed labels)
     6. Data pipeline bugs (bad joins, unit changes, silent nulls)
     7. Not enough (relevant) data
     8. Training/serving skew
     9. No monitoring -> silent decay from drift
    10. No path to production (no owner, no integration, no MLOps)
    11. Stakeholders don't trust or can't act on the output
    12. Ethics / compliance / privacy blockers found too late

LEVELS
    EASY   : A project health scorecard - score a real-looking project, get a risk report.
    MEDIUM : Watch three failures happen in code: no baseline, noisy labels, a bad join.
    HARD   : Silent decay - 12 months in production with vs without monitoring
             and retraining, measured in dollars; then auto-generate a PRE-MORTEM.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_lab_utils import (HAS_MPL, OUT_DIR, RANDOM_STATE, TELCO_CATEGORICAL, TELCO_NUMERIC,
                          TELCO_TARGET, banner, make_telco_churn, note, plt, save_fig, show, step)

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): project health scorecard
# ============================================================================
banner("Project health scorecard", "EASY")
CHECKS = [
    # (question, weight, failure reason it protects against)
    ("Is there a named business owner who will act on predictions?", 3, "No decision attached"),
    ("Is the decision + action written down (who does what with the score)?", 3, "Unclear problem"),
    ("Is the success metric tied to money / a business KPI?", 2, "Wrong metric"),
    ("Is there a simple baseline (rule or heuristic) to beat?", 2, "No baseline"),
    ("Were all features checked for availability at prediction time?", 3, "Leakage / skew"),
    ("Is label quality measured (agreement rate, audit sample)?", 2, "Noisy labels"),
    ("Are data validation checks automated in the pipeline?", 2, "Pipeline bugs"),
    ("Is there an evaluation on out-of-time data?", 2, "Optimistic offline results"),
    ("Is monitoring (drift, performance) planned before launch?", 2, "Silent decay"),
    ("Is there an engineering path to production (API/batch, owner, on-call)?", 3, "Stuck in notebook"),
    ("Have legal/privacy/fairness reviewed the use case?", 2, "Late compliance blocker"),
    ("Will a holdout / A-B test measure real business impact?", 2, "Unproven value"),
]

# A realistic project: a strong data scientist, weak organisation around them
project_answers = [True, False, False, True, False, False, True, False, False, False, True, False]

score = sum(w for (q, w, _), a in zip(CHECKS, project_answers) if a)
max_score = sum(w for _, w, _ in CHECKS)
for (q, w, risk), a in zip(CHECKS, project_answers):
    print(f"  [{'x' if a else ' '}] (w={w}) {q}" + ("" if a else f"   <- RISK: {risk}"))
show("Health score", f"{score}/{max_score} ({score / max_score:.0%})")
verdict = "GREEN - proceed" if score / max_score >= 0.8 else \
          "AMBER - fix gaps before building more models" if score / max_score >= 0.5 else \
          "RED - stop modelling, fix the project setup"
show("Verdict", verdict)
note("""Notice: NOTHING on this list is about the algorithm. The model can be
excellent and the project still fail. Run this checklist at kickoff and
before every launch.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): three failures you can watch happen
# ============================================================================
banner("Failure demos: no baseline, noisy labels, a bad join", "MEDIUM")
df = make_telco_churn(n=10_000)
FEATURES = TELCO_NUMERIC + TELCO_CATEGORICAL
train, test = train_test_split(df, test_size=0.3, stratify=df[TELCO_TARGET], random_state=RANDOM_STATE)
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), TELCO_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), TELCO_CATEGORICAL),
])


def fit_auc(train_df, test_df, clf=None, label=TELCO_TARGET) -> float:
    clf = clf or LogisticRegression(max_iter=2000)
    pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(train_df[FEATURES], train_df[label])
    return roc_auc_score(test_df[TELCO_TARGET], pipe.predict_proba(test_df[FEATURES])[:, 1])


step("Failure 1 - 'Our boosted model has a great AUC!' ... compared to what?")
rule_score = (test.contract_type.eq("Month-to-month").astype(int) * 2
              + test.tenure_months.lt(12).astype(int) + test.internet_service.eq("Fiber optic").astype(int))
show("Business rule (3 lines of SQL) AUC", roc_auc_score(test[TELCO_TARGET], rule_score))
show("Gradient boosting AUC", fit_auc(train, test, HistGradientBoostingClassifier(random_state=RANDOM_STATE)))
note("""The 3-line rule already captures most of the signal. The model's extra lift
might still be worth it - or might not justify a pipeline, monitoring and an
on-call rotation. You can only decide if the baseline is on the slide.""")

step("Failure 2 - noisy labels (e.g. 'churn' tagged inconsistently by different teams)")
for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
    noisy = train.copy()
    flip = rng.random(len(noisy)) < noise
    noisy["noisy_label"] = np.where(flip, 1 - noisy[TELCO_TARGET], noisy[TELCO_TARGET])
    print(f"  {noise:>4.0%} of training labels wrong -> test AUC {fit_auc(noisy, test, label='noisy_label'):.3f}")
note("""Models are surprisingly robust to RANDOM noise - but noise is rarely
random. Systematic label errors (e.g. one region never records churn) teach
the model a wrong pattern. Audit a labelled sample; measure agreement between
labellers before blaming the algorithm.""")

step("Systematic label error: one region's CRM never records cancellations")
biased = train.copy()
biased.loc[biased.region.eq("West"), TELCO_TARGET] = 0
pipe_b = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]).fit(biased[FEATURES], biased[TELCO_TARGET])
west = test.region.eq("West")
biased_test = test[TELCO_TARGET].where(~west, 0)          # the team evaluates with the SAME broken CRM
p_b = pipe_b.predict_proba(test[FEATURES])[:, 1]
show("AUC the team SEES (same broken CRM labels)", roc_auc_score(biased_test, p_b))
show("AUC in REALITY (true labels)", roc_auc_score(test[TELCO_TARGET], p_b))
show("Avg predicted churn - West", pipe_b.predict_proba(test[west][FEATURES])[:, 1].mean())
show("Actual churn - West", test[west][TELCO_TARGET].mean())
note("""The model learned 'West customers never churn' and will never send the
retention team there. Worse: the team evaluates with labels from the same
broken CRM, so their dashboard shows a HEALTHY AUC. A biased label source
corrupts training AND evaluation at once. Evaluate per segment, and audit
labels against an independent source (e.g. billing stop dates).""")

step("Failure 3 - a bad join silently scrambles a key feature")
broken = test.copy()
# A join on a non-unique key re-ordered contract data relative to customers:
broken["contract_type"] = broken.contract_type.sample(frac=1, random_state=1).to_numpy()
pipe_ok = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]).fit(train[FEATURES], train[TELCO_TARGET])
show("AUC with correct join", roc_auc_score(test[TELCO_TARGET], pipe_ok.predict_proba(test[FEATURES])[:, 1]))
show("AUC after the broken join", roc_auc_score(broken[TELCO_TARGET], pipe_ok.predict_proba(broken[FEATURES])[:, 1]))
show("Row count after join (unchanged!)", len(broken))
show("contract_type distribution changed?", not broken.contract_type.value_counts().equals(test.contract_type.value_counts()))
note("""Same rows, same columns, same value distribution - every simple check
passes. Only the row-level ALIGNMENT is wrong. Defend with key-uniqueness
checks before joins and with known-answer tests (score 50 customers whose
values you know).""")

# %% =========================================================================
# LEVEL 3 (HARD): silent decay in production - measured in dollars
# ============================================================================
banner("12 months in production: no monitoring vs monitoring + retraining", "HARD")


def month_batch(month: int, n: int = 3000) -> pd.DataFrame:
    """The world changes gradually: from month 4 a competitor launches a fibre
    network and actively poaches our LOYAL, long-tenure DSL customers. Over 6
    months, the old pattern ("new month-to-month customers churn") fades and a
    new one ("long-tenure DSL customers churn") takes over."""
    b = make_telco_churn(n=n, seed=500 + month)
    d = min(1.0, max(0, month - 3) / 6)            # 0 until month 3, ramps to 1 by month 9
    logit = (-1.4 - 1.2 * d
             + 1.2 * (1 - d) * b.contract_type.eq("Month-to-month")
             + (-0.035 + 0.06 * d) * b.tenure_months   # tenure effect REVERSES
             + 0.015 * (b.monthly_charges - 65)
             + 0.4 * b.internet_service.eq("Fiber optic")
             + 1.5 * d * b.internet_service.eq("DSL")  # brand-new churn driver
             + 0.3 * b.support_tickets_90d)
    b[TELCO_TARGET] = np.random.default_rng(900 + month).binomial(1, 1 / (1 + np.exp(-logit)))
    return b


CALLS, SAVE_RATE, CLV, COST = 400, 0.3, 700, 25
history = month_batch(0, n=8000)
static_model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]).fit(history[FEATURES], history[TELCO_TARGET])
live_model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]).fit(history[FEATURES], history[TELCO_TARGET])
deploy_auc = roc_auc_score(history[TELCO_TARGET], live_model.predict_proba(history[FEATURES])[:, 1])
recent = [history]
rows = []
for month in range(1, 13):
    batch = month_batch(month)
    row = {"month": month}
    for name, model in [("static", static_model), ("monitored", live_model)]:
        p = model.predict_proba(batch[FEATURES])[:, 1]
        called = np.argsort(-p)[:CALLS]
        tp = batch[TELCO_TARGET].to_numpy()[called].sum()
        row[f"{name}_auc"] = roc_auc_score(batch[TELCO_TARGET], p)
        row[f"{name}_profit"] = tp * SAVE_RATE * CLV - CALLS * COST
    # Monitoring policy: labels for this month arrive; if AUC fell > 0.02, retrain on the last 3 months
    recent = (recent + [batch])[-3:]
    row["retrained"] = row["monitored_auc"] < deploy_auc - 0.02
    if row["retrained"]:
        window = pd.concat(recent)
        live_model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]).fit(
            window[FEATURES], window[TELCO_TARGET])
        deploy_auc = roc_auc_score(window[TELCO_TARGET], live_model.predict_proba(window[FEATURES])[:, 1])
    rows.append(row)
prod = pd.DataFrame(rows).set_index("month")
print(prod.round(3).to_string())
lost = prod.monitored_profit.sum() - prod.static_profit.sum()
show("Extra profit from monitoring + retraining", f"${lost:,.0f} over 12 months")
note("""The static model never crashed, never threw an error, and its dashboard
stayed green - it just got quietly worse every month. Note: retraining is
triggered one month AFTER the damage, because churn labels arrive late. In
real systems, feature-drift alerts (PSI, Lab 08) buy you that month back.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(prod.index, prod.static_auc, marker="o", label="static model (never retrained)")
    ax.plot(prod.index, prod.monitored_auc, marker="o", label="monitored + retrained")
    for m in prod.index[prod.retrained]:
        ax.axvline(m + 0.5, color="grey", alpha=0.3)
    ax.set(xlabel="month in production", ylabel="ROC-AUC", title="Silent model decay")
    ax.legend()
    save_fig(fig, "13_hard_silent_decay")

step("Auto-generate a PRE-MORTEM for the project scored in Level 1")
missing = [(q, risk) for (q, w, risk), a in zip(CHECKS, project_answers) if not a]
lines = [
    "# Pre-mortem: Telco churn model",
    "",
    "_Imagine it is 12 months from now and this project FAILED. Why?_",
    "",
    f"Health score: **{score}/{max_score}** - {verdict}",
    "",
    "| # | Likely cause of failure | Unanswered question | Mitigation owner | Due |",
    "|---|---|---|---|---|",
]
lines += [f"| {i} | {risk} | {q} | _TBD_ | _TBD_ |" for i, (q, risk) in enumerate(missing, 1)]
lines += ["", "## Evidence from this lab",
          f"- Static model lost ${lost:,.0f} vs a monitored one over 12 months.",
          "- A systematic label gap in one region was invisible in overall AUC.",
          "- A broken join passed row-count and distribution checks."]
path = OUT_DIR / "13_pre_mortem.md"
path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:12]))
print(f"  ... full report saved to outputs/{path.name}")

banner("KEY TAKEAWAYS")
print("""  1. Most ML failures are organisational and data problems, not algorithm problems.
  2. Always show a baseline; always evaluate per segment, not just overall.
  3. Label quality and join correctness deserve tests, like any other code.
  4. Models decay silently - monitoring and retraining are part of the product.
  5. Run a pre-mortem at kickoff: assume failure, list why, assign owners.""")
