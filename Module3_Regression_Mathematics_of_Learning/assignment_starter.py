"""
ASSIGNMENT STARTER - Compare three regression models and justify the best one
=============================================================================

Read ASSIGNMENT.md first. Reference solution for a DIFFERENT problem: 11_assignment_worked_example.py

    python assignment_starter.py

The script runs as-is: PART A (data, requirements, baseline, evaluation harness) and
PART C (scorecard, rubric, memo) are written for you. Your work is PART B:
build three candidate models, analyse residuals, choose weight scenarios, open the
test set ONCE and write the justification. Re-run until every rubric check is PASS.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit

from ml_lab_utils import DATA_DIR, DELIVERY_TARGET, OUT_DIR, banner, make_delivery_orders, note, show, step
from regression_toolkit import ModelScorecard, Requirement, bootstrap_compare, residual_table

RAW_PATH = DATA_DIR / "delivery_orders.csv"
if not RAW_PATH.exists():                       # generated once, then treated as an immutable raw file
    make_delivery_orders().to_csv(RAW_PATH, index=False)
orders = pd.read_csv(RAW_PATH)
TARGET = DELIVERY_TARGET
LEAKY = "pickup_minutes"

# %% =========================================================================
# PART A - Data, requirements, baseline, evaluation harness (done for you)
# ============================================================================
banner("PART A - The brief")
print("""  The product team shows an ETA ('your food arrives in ~32 min') at checkout.
  A late delivery (actual > ETA + 5 min) is the #1 cause of refunds and churn.
  Operations must be able to explain an ETA to a customer who complains.
  The ETA service is called on every checkout page view.""")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["order_time"])
    return df.assign(hour=t.dt.hour, weekday=t.dt.dayofweek, is_weekend=(t.dt.dayofweek >= 5).astype(int))


orders = add_time_features(orders)
cut = pd.to_datetime(orders.order_time).max() - pd.Timedelta(days=30)
is_test = pd.to_datetime(orders.order_time) > cut
train = orders[~is_test].sort_values("order_time").reset_index(drop=True)
_test = orders[is_test].reset_index(drop=True)              # sealed: only open_test_set() touches it
y_tr = train[TARGET].to_numpy()
show("Train / test orders (test = last 30 days)", f"{len(train):,} / {len(_test):,}")
print(train.head(3).T.to_string())

# The requirements are fixed by the product team. You may ADD requirements, but not loosen these.
REQUIREMENTS = [
    Requirement("Accuracy floor", "MAE", threshold=4.0, hard=True, why="Ops target: typical miss under 4 minutes"),
    Requirement("Late promises", "late rate %", threshold=12.0, hard=True,
                why="Share of orders arriving > 5 min after the ETA drives refunds"),
    Requirement("Bad-weather robustness", "snow/clear MAE ratio", threshold=1.5, hard=True,
                why="ETAs must stay credible on snow days (peak demand)"),
    Requirement("Latency", "latency ms", threshold=20.0, hard=True, why="Called on every checkout page view"),
    Requirement("No post-order information", "leak-free", hard=True, higher_is_better=True,
                why="Only information available at checkout may be used"),
    Requirement("Typical error", "MAE", weight=0.30, why="Customer-perceived accuracy"),
    Requirement("Within 5 minutes", "within 5 min %", weight=0.25, higher_is_better=True,
                why="Share of ETAs customers consider 'right'"),
    Requirement("Late promises (soft)", "late rate %", weight=0.25, why="Fewer refunds"),
    Requirement("Explainability", "explainability", weight=0.20, higher_is_better=True,
                why="Ops must explain an ETA to a complaining customer"),
]
ORIGINAL = {(r.name, r.threshold, r.weight) for r in REQUIREMENTS}
card = ModelScorecard("Delivery ETA shown at checkout", REQUIREMENTS)
FOLDS = TimeSeriesSplit(n_splits=5)                          # always train on the past, validate on the future
_state = {"oof": {}, "families": {}, "fitted": {}, "residuals_done": set(), "test_opened": 0, "test": None}


def eta_metrics(y, p, weather):
    e = np.asarray(y) - np.asarray(p)
    w = np.asarray(weather)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": float(1 - np.sum(e ** 2) / np.sum((np.asarray(y) - np.mean(y)) ** 2)),
            "within 5 min %": float(np.mean(np.abs(e) <= 5) * 100), "late rate %": float(np.mean(e > 5) * 100),
            "snow/clear MAE ratio": float(np.mean(np.abs(e[w == "snow"])) / np.mean(np.abs(e[w == "clear"])))}


def evaluate(name: str, model, explainability: float, why: str = "") -> dict:
    """Time-series CV, latency and a leakage test for one candidate. Adds it to the scorecard.
    explainability: 1.0 = per-feature effects readable (linear), 0.5 = a few rules (shallow tree), 0 = opaque."""
    oof = np.full(len(train), np.nan)
    for tr_i, va_i in FOLDS.split(train):
        m = clone(model).fit(train.iloc[tr_i], y_tr[tr_i])
        oof[va_i] = m.predict(train.iloc[va_i])
    mask = ~np.isnan(oof)
    res = eta_metrics(y_tr[mask], oof[mask], train.weather[mask])
    full = clone(model).fit(train, y_tr)
    one = train.iloc[[0]]
    lat = []
    for _ in range(30):
        t0 = time.perf_counter()
        full.predict(one)
        lat.append((time.perf_counter() - t0) * 1000)
    probe = train.sample(500, random_state=0)
    shuffled = probe.assign(**{LEAKY: probe[LEAKY].sample(frac=1, random_state=1).to_numpy()})
    res["leak-free"] = bool(np.allclose(full.predict(probe), full.predict(shuffled)))
    res["latency ms"] = float(np.median(lat))
    res["explainability"] = float(explainability)
    card.add(name, res, note=why)
    _state["oof"][name] = (oof, mask)
    _state["fitted"][name] = (model, full)
    est = full[-1] if hasattr(full, "steps") else full
    est = getattr(est, "regressor_", getattr(est, "best_estimator_", est))
    _state["families"][name] = type(est).__module__.split(".")[1] if "." in type(est).__module__ else type(est).__name__
    show(f"evaluated: {name}", f"CV MAE {res['MAE']:.2f} min, late {res['late rate %']:.1f}%, "
                               f"leak-free {res['leak-free']}")
    return res


def residual_report(name: str) -> pd.DataFrame:
    """Out-of-fold residuals of one candidate by weather and by distance band."""
    oof, mask = _state["oof"][name]
    d = train[mask]
    seg = d.weather + " / " + pd.cut(d.distance_km, [0, 2, 5, 100], labels=["<2 km", "2-5 km", ">5 km"]).astype(str)
    tab = residual_table(d[TARGET], oof[mask], seg)
    _state["residuals_done"].add(name)
    print(tab.round(2).to_string())
    return tab


def open_test_set() -> pd.DataFrame:
    """Fit every candidate on all training data and score it on the sealed test set. Call ONCE."""
    _state["test_opened"] += 1
    rows = {}
    for name, (model, full) in _state["fitted"].items():
        rows[name] = eta_metrics(_test[TARGET], full.predict(_test), _test.weather)
    tab = pd.DataFrame(rows).T
    _state["test"] = tab
    print(tab.round(2).to_string())
    return tab


class OpsRuleOfThumb(RegressorMixin, BaseEstimator):
    """BASELINE used by ops today: a kitchen-time table per restaurant type + 3 minutes per km."""

    def fit(self, X, y):
        self.kitchen_ = pd.Series(np.asarray(y) - 3 * X["distance_km"].to_numpy()).groupby(
            X["restaurant_type"].to_numpy()).median()
        return self

    def predict(self, X):
        return X["restaurant_type"].map(self.kitchen_).fillna(self.kitchen_.median()).to_numpy() + 3 * X["distance_km"].to_numpy()


step("A1  Baseline")
evaluate("Baseline: ops rule of thumb", OpsRuleOfThumb(), explainability=1.0, why="What ops uses today")

# %% =========================================================================
# PART B - YOUR WORK
# ============================================================================
banner("PART B - Three candidates, residuals, decision (your work)")

step("B1  Three candidate models")
# TODO: build THREE candidates from at least TWO algorithm families (e.g. a regularised linear model,
#       a linear model with domain features, a decision tree). Every candidate is a scikit-learn
#       Pipeline that takes the RAW `train` DataFrame (ColumnTransformer inside!).
#       Ask the leakage question for every column: "is this known at checkout?"
#       Ideas: is_rush_hour, distance x weather, a quantile loss to cut late promises (Lab 02).
# Example of the expected style (uncomment, then build your own three):
# from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import RidgeCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# NUM = ["items", "kitchen_queue", "distance_km", "courier_experience_months", "hour", "is_weekend"]
# CAT = ["restaurant_type", "vehicle", "weather"]
# ridge = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), NUM),
#                                               ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])),
#                   ("model", RidgeCV(alphas=np.logspace(-2, 3, 20)))])
# evaluate("M1 Ridge", ridge, explainability=1.0, why="Readable coefficients in minutes")

step("B2  Residual analysis")
# TODO: call residual_report("<name>") for at least your recommended model. Where does it fail
#       (snow? long trips?) and what did you change because of it? Write it in the justification.

step("B3  Weight scenarios for the sensitivity analysis")
# TODO: at least TWO alternative weightings of the four soft criteria, in this order:
#       [Typical error, Within 5 minutes, Late promises (soft), Explainability]
SCENARIOS = {
    # "accuracy-first": [0.5, 0.3, 0.15, 0.05],
    # "refund-first":   [0.2, 0.2, 0.5, 0.1],
}

step("B4  Open the test set ONCE (after your choice is made on CV)")
# TODO: test_table = open_test_set()

step("B5  Justification")
# TODO: >= 120 words. Name the recommended model, cite numbers (CV and test), state which hard gates
#       decided, what the residuals showed, what the sensitivity analysis showed, and the limitations.
JUSTIFICATION = """
"""

# %% =========================================================================
# PART C - Scorecard, rubric and memo (done for you)
# ============================================================================
banner("PART C - Decide, self-check, write the memo")
board = card.evaluate()
print(board.to_string(float_format=lambda v: f"{v:,.2f}"))
winner = card.recommend()
show("Recommended by the scorecard", winner)

sens = {"as written": winner}
for label, w in SCENARIOS.items():
    reqs = [Requirement(**r.__dict__) for r in REQUIREMENTS]
    for r, wt in zip([r for r in reqs if not r.hard], w):
        r.weight = wt
    sens[label] = ModelScorecard(card.problem, reqs, results=card.results).recommend()
print(pd.Series(sens, name="recommended").to_string())

ci_text = ""
if _state["test"] is not None and len(board) >= 2:
    a, b = board.index[0], board.index[1]
    ci = bootstrap_compare(_test[TARGET], _state["fitted"][a][1].predict(_test), _state["fitted"][b][1].predict(_test),
                           metric=lambda y, p: float(np.mean(np.abs(y - p))))
    ci_text = f"Test MAE({b}) - MAE({a}) = {ci['diff (B - A)']:.2f} min, 95% CI [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"
    show("Top-2 difference on test", ci_text)

candidates = [n for n in card.results if not n.startswith("Baseline")]
words = JUSTIFICATION.split()
numbers = [w for w in words if any(ch.isdigit() for ch in w)]
checks = {
    ">= 3 candidates besides the baseline (6 pts)": len(candidates) >= 3,
    ">= 2 algorithm families (4 pts)": len({_state["families"][n] for n in candidates}) >= 2,
    "No candidate uses post-order information (6 pts)": bool(candidates) and all(
        card.results[n]["leak-free"] for n in candidates),
    "Requirements not loosened (4 pts)": ORIGINAL <= {(r.name, r.threshold, r.weight) for r in REQUIREMENTS},
    "Residuals analysed for the recommended model (4 pts)": winner in _state["residuals_done"],
    ">= 2 weight scenarios in the sensitivity analysis (4 pts)": len(SCENARIOS) >= 2,
    "Test set opened exactly once (4 pts)": _state["test_opened"] == 1,
    "Justification: >= 120 words, names the winner, cites >= 3 numbers (8 pts)": len(words) >= 120
    and winner is not None and winner in JUSTIFICATION and len(numbers) >= 3,
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
points = sum(int(k.split("(")[-1].split()[0]) for k, v in checks.items() if v)
show("Automated rubric points", f"{points} / 40")

if all(checks.values()):
    card.to_markdown(OUT_DIR / "assignment_eta_memo.md", justification=JUSTIFICATION.strip() + f"\n\n{ci_text}",
                     header=f"Train {len(train):,} orders (TimeSeriesSplit, 5 folds); test = last 30 days "
                            f"({len(_test):,} orders, opened once). Sensitivity: {sens}")
    card.to_json(OUT_DIR / "assignment_eta_scorecard.json")
    print("  [saved] outputs/assignment_eta_memo.md + outputs/assignment_eta_scorecard.json")
else:
    note("""Not all checks pass yet - keep going with PART B. The memo is written
automatically once every check is PASS.""")
