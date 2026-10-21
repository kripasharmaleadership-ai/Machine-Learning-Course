"""
LAB 09 - Converting a Business Problem into an ML Problem
=========================================================

Business people never say "please train a binary classifier". They say:
    "Churn is killing us."   "Fraud losses doubled."   "We keep running out of stock."

A principal engineer's job is to TRANSLATE that into something a model can learn:

    Business goal        : Reduce churn-driven revenue loss by 10% this year
    Decision to support  : Which customers should the retention team call each week?
    Prediction target    : P(customer cancels within the next 90 days)
    Unit of observation  : one active customer at a weekly snapshot date
    Features             : only data available BEFORE the snapshot date
    Action               : top-N customers -> call list -> offer
    Success metric       : revenue saved (business) <- precision@N (model) <- AUC (dev)

LEVELS
    EASY   : "Does this even need ML?" - a screening checklist applied to 6 requests.
    MEDIUM : The ML Problem Framing Canvas for 3 enterprise problems, and why one
             goal can be framed many ways.
    HARD   : (a) Build a POINT-IN-TIME correct training set from a raw event log
             (the label definition is code, and getting it wrong leaks the future).
             (b) The framing that actually matters: predict CHURN or predict who
             the OFFER WILL SAVE? An uplift model beats a churn model at the real goal.
"""
# %%
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ml_lab_utils import RANDOM_STATE, banner, note, show, step

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): Does this problem even need ML?
# ============================================================================
banner("Screening: is ML the right tool?", "EASY")


@dataclass
class Request:
    name: str
    rules_can_solve_it: bool      # can a human write down the exact logic?
    pattern_in_data: bool         # is there a learnable relationship?
    has_historical_data: bool     # do we have (or can we get) examples with outcomes?
    decision_repeats_often: bool  # made many times (scale justifies ML)?
    tolerates_errors: bool        # is an occasional wrong prediction acceptable?


def ml_verdict(r: Request) -> str:
    if r.rules_can_solve_it:
        return "NO ML - write the rules (cheaper, exact, auditable)"
    if not r.pattern_in_data:
        return "NO ML - nothing to learn (randomness or missing signal)"
    if not r.has_historical_data:
        return "NOT YET - collect labelled data first (or start with heuristics)"
    if not r.decision_repeats_often:
        return "PROBABLY NOT - one-off decision, use analysis not a model"
    if not r.tolerates_errors:
        return "ML + HUMAN REVIEW - model assists, human decides"
    return "GOOD ML CANDIDATE"


requests = [
    Request("Calculate VAT on invoices", True, True, True, True, False),
    Request("Predict which customers will churn", False, True, True, True, True),
    Request("Predict tomorrow's lottery numbers", False, False, True, True, True),
    Request("Detect fraud in a brand-new product (no history)", False, True, False, True, True),
    Request("Choose the location of our one new HQ", False, True, True, False, True),
    Request("Flag tumours on radiology scans", False, True, True, True, False),
]
for r in requests:
    print(f"  {r.name:<50} -> {ml_verdict(r)}")
note("""The best ML engineers say 'no ML needed' often. ML adds cost: data
pipelines, monitoring, retraining, explainability. Use it when the pattern is
real, too complex for rules, backed by data, and repeated at scale.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): The ML Problem Framing Canvas
# ============================================================================
banner("ML Problem Framing Canvas - 3 enterprise problems", "MEDIUM")


@dataclass
class FramingCanvas:
    business_problem: str
    business_goal: str
    decision: str
    ml_task: str
    target_definition: str
    unit_of_observation: str
    prediction_time: str
    features_available: str
    action_on_prediction: str
    business_metric: str
    model_metric: str
    baseline: str
    risks: str

    def print_card(self):
        width = max(len(k) for k in self.__dict__)
        print("  +" + "-" * 100)
        for k, v in self.__dict__.items():
            print(f"  | {k.replace('_', ' ').upper():<{width}} : {v}")
        print("  +" + "-" * 100)


canvases = [
    FramingCanvas(
        "Churn is killing us",
        "Cut churn-driven revenue loss 10% in FY27",
        "Who does the retention team call this week? (capacity: 500 calls/week)",
        "Binary classification (ranked by probability)",
        "Customer cancels contract within 90 days after the snapshot date",
        "Active customer x weekly snapshot",
        "Every Monday 06:00, batch",
        "Tenure, contract, charges, usage trend, tickets (all BEFORE snapshot)",
        "Top-500 by expected value -> call + offer",
        "Revenue retained vs control group",
        "Precision@500, PR-AUC",
        "Rule: month-to-month + tenure < 6 months",
        "Leakage from post-cancel fields; offers to customers who'd stay anyway",
    ),
    FramingCanvas(
        "Card fraud losses doubled",
        "Reduce fraud losses 30% without hurting good customers",
        "Approve / step-up (OTP) / decline each card transaction",
        "Binary classification, real-time, extreme imbalance (0.1%)",
        "Transaction confirmed as fraud by chargeback within 60 days",
        "Single card transaction",
        "Real-time, < 50 ms per transaction",
        "Amount, merchant, device, geo-velocity, card history up to that second",
        "Score > t1: OTP challenge; score > t2: decline",
        "Fraud $ prevented minus friction cost",
        "Recall at 0.5% false-positive rate, PR-AUC",
        "Existing rules engine",
        "Label delay (60 days); adversaries adapt; customer friction",
    ),
    FramingCanvas(
        "We keep running out of stock",
        "Cut stock-outs 25% while keeping inventory cost flat",
        "How many units to order per store x product each week",
        "Regression / time-series forecasting (with prediction intervals)",
        "Units DEMANDED next week (careful: sales are capped by stock = censored)",
        "Store x product x week",
        "Sunday night, batch, 2 weeks ahead",
        "Past sales, promotions, price, holidays, weather forecast",
        "Order quantity = forecast quantile matching service level",
        "Stock-out rate, inventory holding cost",
        "MAE / WAPE, quantile (pinball) loss",
        "Same week last year; 4-week moving average",
        "Censored demand, new products with no history, promotions",
    ),
]
for c in canvases:
    c.print_card()
note("""Notice how much is decided BEFORE any code: exact label definition, when
the prediction is made, which data exists at that moment, what action
follows, and which metric links to money. Most failed projects skipped this
page. Also note the stock-out trap: you cannot learn DEMAND from SALES when
shelves were empty - the label itself was wrong.""")

# %% =========================================================================
# LEVEL 3 (HARD a): Point-in-time correct training data from raw events
# ============================================================================
banner("From raw event logs to a point-in-time correct training set", "HARD")
note("""Real companies do not have a tidy 'churned' column. They have EVENT LOGS:
logins, payments, support calls, cancellations - with timestamps. The label
and features must be built so that, for every snapshot date, features use
ONLY the past and the label uses ONLY the future window.""")

N_CUST = 3000
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-30")
signup = START + pd.to_timedelta(rng.integers(0, 540, N_CUST), unit="D")
engagement = rng.lognormal(0, 0.5, N_CUST)
weekly_cancel_p = np.clip(0.006 / engagement, 0.001, 0.05)
life_weeks = rng.geometric(weekly_cancel_p)
cancel_date = signup + pd.to_timedelta(life_weeks * 7, unit="D")
cancel_date = pd.Series(cancel_date).where(cancel_date <= END)        # NaT = still active
customers = pd.DataFrame({"customer_id": np.arange(N_CUST), "signup": signup, "cancel_date": cancel_date})

# Weekly activity events. Usage DECLINES and support calls RISE in the 8 weeks before cancelling.
last_day = cancel_date.fillna(END)
n_weeks = ((last_day - signup).dt.days // 7).clip(lower=1).to_numpy()
cid = np.repeat(np.arange(N_CUST), n_weeks)
week_idx = np.concatenate([np.arange(k) for k in n_weeks])
weeks_to_cancel = np.where(customers.cancel_date.notna().to_numpy()[cid], n_weeks[cid] - week_idx, 999)
decline = np.clip(weeks_to_cancel / 8, 0.15, 1)
events = pd.DataFrame({
    "customer_id": cid,
    "week_start": signup[cid] + pd.to_timedelta(week_idx * 7, unit="D"),
    "logins": rng.poisson(4 * engagement[cid] * decline),
    "support_calls": rng.poisson(0.08 + 0.35 * (weeks_to_cancel <= 6)),
})
show("Customers", N_CUST)
show("Weekly event rows", f"{len(events):,}")
print(events.head(4).to_string(index=False))


def build_snapshot(snapshot: pd.Timestamp, horizon_days: int = 90, point_in_time: bool = True) -> pd.DataFrame:
    """One training row per customer ACTIVE at `snapshot`.
    Features: activity in the 90 days BEFORE the snapshot.
    Label   : cancelled within (snapshot, snapshot + horizon]."""
    active = customers[(customers.signup <= snapshot) &
                       (customers.cancel_date.isna() | (customers.cancel_date > snapshot))]
    window_start = snapshot - pd.Timedelta(days=90)
    if point_in_time:
        ev = events[(events.week_start >= window_start) & (events.week_start < snapshot)]
    else:
        # THE BUG: 'just use the latest activity table' - includes weeks AFTER the snapshot
        ev = events[events.week_start >= window_start]
    feats = ev.groupby("customer_id").agg(logins_90d=("logins", "sum"),
                                         support_calls_90d=("support_calls", "sum"),
                                         active_weeks=("logins", lambda s: (s > 0).sum()))
    out = active.set_index("customer_id").join(feats).fillna(
        {"logins_90d": 0, "support_calls_90d": 0, "active_weeks": 0})
    out["tenure_days"] = (snapshot - out.signup).dt.days
    out["label_churn_90d"] = (out.cancel_date.notna() &
                              (out.cancel_date <= snapshot + pd.Timedelta(days=horizon_days))).astype(int)
    out["snapshot"] = snapshot
    return out.reset_index()


FEATS = ["logins_90d", "support_calls_90d", "active_weeks", "tenure_days"]
train_snaps = pd.date_range("2025-01-01", "2025-09-01", freq="MS")
test_snap = pd.Timestamp("2025-12-01")

step("Label definition changes the problem - churn rate by horizon")
for h in [30, 90, 180]:
    print(f"  horizon {h:>3} days -> churn rate {build_snapshot(test_snap, h).label_churn_90d.mean():.1%}")
note("""'Churn' has no natural definition: 30-day churn and 180-day churn are
different problems with different features and actions. Agree on it with the
business and write it down (the canvas above).""")

step("Train on monthly snapshots Jan-Sep 2025, test on the Dec 2025 snapshot")
results = {}
for pit in [True, False]:
    train_set = pd.concat([build_snapshot(s, point_in_time=pit) for s in train_snaps])
    offline_test = build_snapshot(test_snap, point_in_time=pit)
    production_test = build_snapshot(test_snap, point_in_time=True)   # production never sees the future
    model = HistGradientBoostingClassifier(max_depth=3, random_state=RANDOM_STATE)
    model.fit(train_set[FEATS], train_set.label_churn_90d)
    tag = "point-in-time (correct)" if pit else "latest table (LEAKY)"
    results[tag] = {
        "train rows": len(train_set),
        "offline test AUC": roc_auc_score(offline_test.label_churn_90d, model.predict_proba(offline_test[FEATS])[:, 1]),
        "real production AUC": roc_auc_score(production_test.label_churn_90d,
                                             model.predict_proba(production_test[FEATS])[:, 1]),
    }
res = pd.DataFrame(results).T
res["train rows"] = res["train rows"].astype(int)
print(res.round(3).to_string())
note("""The leaky pipeline counted logins that happened AFTER the snapshot - after
some customers had already stopped using the service. Offline it looks
brilliant. In production that future data does not exist yet, and the
model, trained to rely on it, performs worse than the honest one. This bug
is invisible in a random train/test split because BOTH sides leak.""")

# %% =========================================================================
# LEVEL 3 (HARD b): Predict churn, or predict who the offer will SAVE?
# ============================================================================
banner("The framing that matters: churn model vs uplift model", "HARD")
note("""The business goal is not 'find churners' - it is 'SAVE customers with
offers'. Four kinds of customers exist:
  Sure things      : stay anyway           -> offer is wasted money
  Lost causes      : leave anyway          -> offer is wasted money
  Persuadables     : stay ONLY with offer  -> the only ones worth targeting
  Sleeping dogs    : offer reminds them to leave -> offer HURTS
A churn model ranks by risk, which is full of lost causes. An UPLIFT model
estimates the EFFECT of the offer per customer.""", keep_lines=True)

n = 20_000
risk_signal = rng.normal(0, 1, n)          # drives baseline churn
price_sensitive = rng.normal(0, 1, n)      # drives response to a discount offer
X_up = pd.DataFrame({"risk_signal": risk_signal, "price_sensitive": price_sensitive,
                     "noise": rng.normal(0, 1, n)})
p_churn_no_offer = 1 / (1 + np.exp(-(-1.0 + 1.5 * risk_signal)))
# Offer helps price-sensitive, medium-risk customers; does little for extreme risk.
effect = 0.25 * (1 / (1 + np.exp(-2 * price_sensitive))) * np.exp(-0.5 * risk_signal ** 2)
p_churn_offer = np.clip(p_churn_no_offer - effect, 0, 1)

# A past RANDOMIZED campaign: 50% got the offer (this is how uplift data is collected)
treated = rng.binomial(1, 0.5, n)
churned = rng.binomial(1, np.where(treated == 1, p_churn_offer, p_churn_no_offer))

# Churn model (standard approach): trained on control group
churn_model = LogisticRegression().fit(X_up[treated == 0], churned[treated == 0])
# Uplift model (T-learner): one model per arm, uplift = P(churn|no offer) - P(churn|offer)
m_control = HistGradientBoostingClassifier(max_depth=3, random_state=1).fit(X_up[treated == 0], churned[treated == 0])
m_treated = HistGradientBoostingClassifier(max_depth=3, random_state=1).fit(X_up[treated == 1], churned[treated == 1])
uplift_score = m_control.predict_proba(X_up)[:, 1] - m_treated.predict_proba(X_up)[:, 1]
churn_score = churn_model.predict_proba(X_up)[:, 1]

budget = int(0.2 * n)   # we can afford offers for 20% of customers
rows = []
for name, score in [("random targeting", rng.random(n)),
                    ("target highest CHURN risk", churn_score),
                    ("target highest UPLIFT", uplift_score)]:
    chosen = np.argsort(-score)[:budget]
    rows.append({"strategy": name,
                 "avg churn risk of targeted": p_churn_no_offer[chosen].mean(),
                 "customers actually saved": round(effect[chosen].sum())})
print(pd.DataFrame(rows).round(3).to_string(index=False))
note("""Targeting by churn risk is WORSE THAN RANDOM here: it picks the riskiest
customers, and many are lost causes the offer can't save. The uplift model picks customers the offer CHANGES.
Same data, same budget, better business result - purely from framing the
problem correctly. (The 'saved' column uses the simulation's ground truth;
in real life you measure it with a holdout control group.)""")

banner("KEY TAKEAWAYS")
print("""  1. Start from the DECISION the business makes, not from the data or the algorithm.
  2. Fill a framing canvas: target definition, unit, prediction time, action, metrics, baseline.
  3. The label is code: build it point-in-time from events, or you will leak the future.
  4. 'Who will churn?' and 'Who can we save?' are different ML problems.
  5. Sometimes the right answer is 'no ML' - and that is a senior answer.""")
