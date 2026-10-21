"""
LAB 08 - The End-to-End ML Lifecycle
====================================

THE LIFECYCLE (it is a LOOP, not a line)

   1. Business problem  ->  2. Data collection  ->  3. Data validation & EDA
          ^                                                  |
          |                                                  v
   9. Retrain / retire  <-  8. Monitor  <-  7. Deploy  <-  4. Feature prep
                                               ^             |
                                               |             v
                            6. Evaluate & select  <-  5. Train (baseline first!)

   Rule of thumb from industry: model training is ~10-20% of the work.
   The rest is data, evaluation, deployment, monitoring - the parts tutorials skip.

USE CASE THROUGHOUT: "Reduce customer churn at TelcoCo"

LEVELS
    EASY   : The whole loop in ~40 lines (understand the shape).
    MEDIUM : Do it properly - data validation, pipelines, model comparison with CV,
             a business-driven threshold, one final test.
    HARD   : MLOps - model registry + model card, a prediction service, production
             monitoring with drift (PSI), a retraining trigger, and a
             champion/challenger promotion decision.
"""
# %%
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_lab_utils import (OUT_DIR, RANDOM_STATE, TELCO_CATEGORICAL, TELCO_LEAKY_COLUMNS,
                          TELCO_NUMERIC, TELCO_TARGET, banner, make_telco_churn, note, show, step)

FEATURES = TELCO_NUMERIC + TELCO_CATEGORICAL

# %% =========================================================================
# LEVEL 1 (EASY): the whole lifecycle in ~40 lines
# ============================================================================
banner("The ML lifecycle, minimal version", "EASY")

step("1. Business problem")
print("  'Predict which customers will churn in the next 90 days, so retention can call them.'")

step("2. Collect data")
df = make_telco_churn()
show("Customers / churn rate", f"{len(df)} / {df[TELCO_TARGET].mean():.1%}")

step("3. Prepare data (drop IDs & leaky columns, split)")
X, y = df[FEATURES], df[TELCO_TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                    random_state=RANDOM_STATE)

step("4. Train a BASELINE, then a real model")
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), TELCO_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), TELCO_CATEGORICAL),
])
baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]).fit(X_train, y_train)

step("5. Evaluate on unseen data")
show("Baseline ROC-AUC", roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1]))
show("Model ROC-AUC", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

step("6. Use it: score customers, hand the top list to the retention team")
top = X_test.assign(churn_risk=model.predict_proba(X_test)[:, 1]).nlargest(5, "churn_risk")
print(top[["tenure_months", "contract_type", "monthly_charges", "churn_risk"]].round(3))
note("""That's the shape of every ML project. Now let's see what a real team adds
at each step - because this minimal version would not survive production.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): doing each step properly
# ============================================================================
banner("Lifecycle done properly", "MEDIUM")

step("Step 3a - DATA VALIDATION (automated checks, fail fast)")


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("no duplicate customer_id", frame.customer_id.is_unique),
        ("target is binary", set(frame[TELCO_TARGET].unique()) <= {0, 1}),
        ("churn rate between 5% and 60%", 0.05 <= frame[TELCO_TARGET].mean() <= 0.60),
        ("tenure within 0..120 months", frame.tenure_months.between(0, 120).all()),
        ("monthly_charges > 0", (frame.monthly_charges > 0).all()),
        ("missing values < 5% per column", (frame[FEATURES].isna().mean() < 0.05).all()),
        ("all expected columns present", set(FEATURES + [TELCO_TARGET]) <= set(frame.columns)),
    ]
    return pd.DataFrame(checks, columns=["check", "passed"])


report = validate_dataset(df)
print(report.to_string(index=False))
assert report.passed.all(), "Data validation failed - stop the pipeline!"
note("""In production this runs on EVERY new data delivery (tools: Great
Expectations, pandera, TFDV). Bad data stopped here costs minutes; bad
data that reaches a deployed model costs customers.""")

step("Step 3b - Quick EDA: which features relate to churn?")
for col in ["contract_type", "internet_service", "payment_method"]:
    rates = df.groupby(col)[TELCO_TARGET].agg(["mean", "size"]).round(3)
    print(f"\n  churn rate by {col}:\n{rates.to_string()}")

step("Step 4-5 - Compare candidate models with 5-fold CV (on TRAIN only)")
candidates = {
    "baseline (prior)": DummyClassifier(strategy="prior"),
    "logistic regression": LogisticRegression(max_iter=2000),
    "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                            random_state=RANDOM_STATE, n_jobs=-1),
    "hist gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                             random_state=RANDOM_STATE),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, clf in candidates.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_rows.append({"model": name, "cv_auc_mean": s.mean(), "cv_auc_std": s.std()})
cv_table = pd.DataFrame(cv_rows).sort_values("cv_auc_mean", ascending=False)
print(cv_table.round(4).to_string(index=False))
note("""Here the simple logistic regression is as good as the fancy models - the
true churn mechanism is close to linear. When scores tie within the +/- std,
choose the SIMPLER model: faster, cheaper, easier to explain to the business
and to regulators.""")

step("Step 6 - Choose the decision THRESHOLD with business costs (on validation data)")
X_tr2, X_val, y_tr2, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train,
                                              random_state=RANDOM_STATE)
champion = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))]).fit(X_tr2, y_tr2)
p_val = champion.predict_proba(X_val)[:, 1]
COST_OFFER, SAVED_VALUE, SAVE_RATE = 20, 600, 0.30   # $ per offer, $ CLV saved, offer success rate


def profit_at(threshold: float, probs: np.ndarray, actual: np.ndarray) -> float:
    targeted = probs >= threshold
    tp = (targeted & (actual == 1)).sum()
    return tp * SAVE_RATE * SAVED_VALUE - targeted.sum() * COST_OFFER


thresholds = np.round(np.arange(0.05, 0.95, 0.05), 2)
profits = [profit_at(t, p_val, y_val.to_numpy()) for t in thresholds]
best_t = float(thresholds[int(np.argmax(profits))])
for t, pr in zip(thresholds, profits):
    if t in (0.05, 0.15, 0.25, 0.35, 0.5, 0.7, 0.9) or t == best_t:
        print(f"  threshold {t:.2f}: profit on validation = ${pr:>9,.0f}{'   <- best' if t == best_t else ''}")
note(f"""The default 0.5 is NOT optimal. Offers are cheap (${COST_OFFER}) and a saved
customer is worth ${SAVED_VALUE * SAVE_RATE:.0f} in expectation, so it pays to contact more people.
The threshold is a BUSINESS parameter, tuned on validation data.""")

step("Step 6b - Final, one-time evaluation on the TEST set")
champion.fit(X_train, y_train)                         # refit on all training data
p_test = champion.predict_proba(X_test)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_test, p_test >= best_t).ravel()
show("Test ROC-AUC", roc_auc_score(y_test, p_test))
show("Test PR-AUC (average precision)", average_precision_score(y_test, p_test))
show(f"Confusion @ {best_t}: TP / FP / FN / TN", f"{tp} / {fp} / {fn} / {tn}")
show("Expected profit on test customers", f"${profit_at(best_t, p_test, y_test.to_numpy()):,.0f}")

# %% =========================================================================
# LEVEL 3 (HARD): MLOps - registry, serving, monitoring, retraining
# ============================================================================
banner("MLOps: registry -> serve -> monitor -> retrain", "HARD")
REGISTRY = OUT_DIR / "model_registry"
REGISTRY.mkdir(exist_ok=True)


def data_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()[:12]


def register_model(pipe, version: str, train_X, train_y, metrics: dict, threshold: float) -> dict:
    """Save the model + a MODEL CARD: everything needed to audit or reproduce it."""
    card = {
        "model_name": "telco_churn",
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": type(pipe.named_steps["clf"]).__name__,
        "features": FEATURES,
        "excluded_columns": ["customer_id", "signup_date"] + TELCO_LEAKY_COLUMNS,
        "training_rows": int(len(train_X)),
        "training_data_fingerprint": data_fingerprint(train_X),
        "training_churn_rate": round(float(train_y.mean()), 4),
        "decision_threshold": threshold,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "intended_use": "Rank customers for retention outreach. Not for pricing or credit decisions.",
        "owner": "ml-platform@telcoco.example",
    }
    joblib.dump(pipe, REGISTRY / f"telco_churn_{version}.joblib")
    (REGISTRY / f"telco_churn_{version}.json").write_text(json.dumps(card, indent=2))
    return card


card_v1 = register_model(champion, "v1", X_train, y_train,
                         {"test_roc_auc": roc_auc_score(y_test, p_test),
                          "test_pr_auc": average_precision_score(y_test, p_test)}, best_t)
step("Model registered - model card")
print(json.dumps({k: card_v1[k] for k in ["version", "algorithm", "training_rows",
                                          "training_data_fingerprint", "decision_threshold",
                                          "metrics", "intended_use"]}, indent=2))

step("Serving: a prediction function (what sits behind the REST API)")
served_model = joblib.load(REGISTRY / "telco_churn_v1.joblib")


def predict_endpoint(records: list[dict]) -> list[dict]:
    frame = pd.DataFrame(records)[FEATURES]
    probs = served_model.predict_proba(frame)[:, 1]
    return [{"churn_probability": round(float(p), 4), "contact": bool(p >= best_t),
             "model_version": "v1"} for p in probs]


print("  ", predict_endpoint([X_test.iloc[0].to_dict()]))

step("Monitoring: 3 months of production traffic")


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index - the industry-standard drift score.
    < 0.1 stable | 0.1-0.25 moderate shift | > 0.25 major shift (investigate)."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def production_month(month: int) -> pd.DataFrame:
    batch = make_telco_churn(n=2000, seed=100 + month)
    if month == 3:
        # Month 3: price increase hits + a fibre network outage drives churn.
        batch["monthly_charges"] = (batch.monthly_charges * 1.18).round(2)
        fiber = batch.internet_service.eq("Fiber optic")
        flip = fiber & (np.random.default_rng(month).random(len(batch)) < 0.35)
        batch.loc[flip, TELCO_TARGET] = 1
    return batch


train_scores = champion.predict_proba(X_train)[:, 1]
monitor_rows, batches = [], {}
for month in [1, 2, 3]:
    batch = production_month(month)
    batches[month] = batch
    scores = champion.predict_proba(batch[FEATURES])[:, 1]
    monitor_rows.append({
        "month": month,
        "psi_monthly_charges": psi(X_train.monthly_charges.to_numpy(), batch.monthly_charges.to_numpy()),
        "psi_tenure": psi(X_train.tenure_months.to_numpy(), batch.tenure_months.to_numpy()),
        "psi_model_score": psi(train_scores, scores),
        "avg_predicted_churn": scores.mean(),
        "actual_churn (labels arrive later)": batch[TELCO_TARGET].mean(),
        "auc (labels arrive later)": roc_auc_score(batch[TELCO_TARGET], scores),
    })
monitoring = pd.DataFrame(monitor_rows).set_index("month")
print(monitoring.round(3).T.to_string())

step("Retraining trigger (rules agreed with the business up front)")
latest = monitoring.loc[3]
triggers = {
    "feature drift (any PSI > 0.25)": bool(latest.psi_monthly_charges > 0.25 or latest.psi_tenure > 0.25),
    "score drift (score PSI > 0.10)": bool(latest.psi_model_score > 0.10),
    "AUC dropped > 0.03 vs card": bool(card_v1["metrics"]["test_roc_auc"] - latest["auc (labels arrive later)"] > 0.03),
    "calibration gap > 5 pts": bool(abs(latest.avg_predicted_churn - latest["actual_churn (labels arrive later)"]) > 0.05),
}
for k, v in triggers.items():
    print(f"  {'FIRED' if v else 'ok   '} | {k}")
note("""Months 1-2 look healthy. Month 3: the monthly_charges PSI explodes (a price
rise = FEATURE drift) - an early warning that needs NO labels. Notice the
score PSI barely moved: price is only one of many inputs, so watching the
score alone would have missed it. Monitor features AND scores. When labels
arrive ~90 days later, the damage is confirmed: AUC fell and the model
under-estimates churn by ~10 points, because a fibre outage created a churn
driver the model has never seen (CONCEPT drift).""")

step("Champion / challenger: retrain on recent data, promote only if better")
m3 = batches[3]
m3_train, m3_holdout = train_test_split(m3, test_size=0.5, stratify=m3[TELCO_TARGET],
                                        random_state=RANDOM_STATE)
recent = pd.concat([df.sample(2000, random_state=RANDOM_STATE), m3_train])
challenger = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])
challenger.fit(recent[FEATURES], recent[TELCO_TARGET])
champ_auc = roc_auc_score(m3_holdout[TELCO_TARGET], champion.predict_proba(m3_holdout[FEATURES])[:, 1])
chall_auc = roc_auc_score(m3_holdout[TELCO_TARGET], challenger.predict_proba(m3_holdout[FEATURES])[:, 1])
show("Champion v1 AUC on recent holdout", champ_auc)
show("Challenger v2 AUC on recent holdout", chall_auc)
if chall_auc > champ_auc + 0.01:
    register_model(challenger, "v2", recent[FEATURES], recent[TELCO_TARGET],
                   {"recent_holdout_roc_auc": chall_auc}, best_t)
    print("  DECISION: promote v2 (after shadow deployment / A-B test in a real system)")
else:
    print("  DECISION: keep v1 - challenger is not clearly better")
print("  Registry contents:", sorted(p.name for p in REGISTRY.iterdir()))

banner("KEY TAKEAWAYS")
print("""  1. The lifecycle is a LOOP: deploy is the middle, not the end.
  2. Validate data automatically before training; prefer the simplest model that ties.
  3. Thresholds come from business economics, tuned on validation data.
  4. Every model ships with a model card: data fingerprint, metrics, intended use, owner.
  5. Monitor drift (PSI) because labels arrive late; retrain on triggers; promote
     a challenger only when it beats the champion on recent data.""")
