"""
ASSIGNMENT STARTER - Design threshold strategies for sepsis screening
=====================================================================

Read ASSIGNMENT.md first. Reference solution for a DIFFERENT scenario (card fraud):
13_assignment_worked_example_fraud_thresholds.py

    python assignment_starter.py

The script runs as-is: PART A (data, requirements, risk model, baseline, evaluation harness)
and PART C (scorecard, rubric, memo) are written for you. Your work is PART B: design at
least four threshold strategies, audit them by site, stress-test the choice, open the test
set ONCE and write the justification. Re-run until every rubric check is PASS.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_lab_utils import DATA_DIR, OUT_DIR, SEPSIS_TARGET, banner, make_ed_patients, note, show, step
from classification_toolkit import ModelScorecard, Requirement, binary_metrics, segment_report

RAW_PATH = DATA_DIR / "ed_visits.csv"
if not RAW_PATH.exists():                       # generated once, then treated as an immutable raw file
    make_ed_patients().to_csv(RAW_PATH, index=False)
visits = pd.read_csv(RAW_PATH)
TARGET = SEPSIS_TARGET
LEAKY = "antibiotics_within_1h"

# %% =========================================================================
# PART A - Data, requirements, risk model, baseline, harness (done for you)
# ============================================================================
banner("PART A - The brief")
print("""  An emergency department wants an early-warning score for SEPSIS at triage. Every adult
  visit gets a risk score; the screening POLICY decides who triggers a sepsis alert
  (immediate physician review, blood cultures, lactate). A missed sepsis can be fatal;
  every false alert costs ~20 minutes of a physician and, at scale, causes alert fatigue.
  Two hospitals share the system: site A (city) and site B (rural, point-of-care lactate
  is usually NOT available). The clinical safety committee has fixed the requirements.""")

t = pd.to_datetime(visits["arrival_time"])
day = (t - t.min()).dt.days
train = visits[day < 120].reset_index(drop=True)
val = visits[(day >= 120) & (day < 150)].reset_index(drop=True)
_test = visits[day >= 150].reset_index(drop=True)            # sealed: only open_test_set() touches it
show("Train / validation / test visits", f"{len(train):,} / {len(val):,} / {len(_test):,}")
show("Sepsis rate train / validation", f"{train[TARGET].mean():.3f} / {val[TARGET].mean():.3f}")
print(train.drop(columns=["visit_id"]).head(3).T.to_string())

# The requirements are fixed by the committee. You may ADD requirements, but not loosen these.
REQUIREMENTS = [
    Requirement("Sensitivity", "sensitivity %", threshold=85.0, hard=True, higher_is_better=True,
                why="Surviving Sepsis guidance: missing sepsis at triage costs lives"),
    Requirement("Alert budget", "alerts per 100 visits", threshold=20.0, hard=True,
                why="Physician capacity: at most 20 alerts per 100 visits, or alerts get ignored"),
    Requirement("Site parity", "site B / site A sensitivity", threshold=0.90, hard=True, higher_is_better=True,
                why="Equity: rural patients must not be screened worse than city patients"),
    Requirement("Triage-time information only", "leak-free", hard=True, higher_is_better=True,
                why="The alert fires at triage; later information may not be used"),
    Requirement("Missed sepsis", "missed per 1,000 visits", weight=0.40, why="Each miss is a potential death"),
    Requirement("Alert load", "alerts per 100 visits", weight=0.30, why="Alert fatigue and physician time"),
    Requirement("Positive predictive value", "PPV %", weight=0.15, higher_is_better=True,
                why="Clinician trust: share of alerts that are real sepsis"),
    Requirement("Simplicity", "simplicity", weight=0.15, higher_is_better=True,
                why="1 = one threshold to explain and audit; lower = more moving parts"),
]
ORIGINAL = {(r.name, r.threshold, r.weight) for r in REQUIREMENTS}
COST_RATIO = 30          # the committee's estimate: one missed sepsis is as bad as 30 false alerts
KINDS = {"statistical", "fixed-sensitivity", "alert-budget", "cost-based", "segment-aware", "combination", "other"}

step("A1  The risk model (logistic regression on triage information)")
NUM = ["age", "comorbidities", "heart_rate", "resp_rate", "temp_c", "systolic_bp", "spo2", "wbc", "lactate"]
BIN = ["nursing_home", "arrived_by_ambulance", "altered_mental_status"]


def make_risk_model():
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
        ("bin", "passthrough", BIN),
        ("site", OneHotEncoder(drop="first"), ["site"])])
    return make_pipeline(prep, LogisticRegression(max_iter=3000))


MODEL = make_risk_model().fit(train, train[TARGET])           # you MAY replace MODEL in B0 (optional)


def with_risk(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(risk=MODEL.predict_proba(df)[:, 1])


def qsofa(d: pd.DataFrame) -> pd.Series:
    """quick SOFA, the bedside clinical rule: 1 point each for RR >= 22, altered mentation, SBP <= 100."""
    return (d.resp_rate >= 22).astype(int) + d.altered_mental_status + (d.systolic_bp <= 100).astype(int)


_state = {"policies": {}, "results": {}, "kinds": {}, "segments_done": set(), "test_opened": 0, "test": None}
card = ModelScorecard("Sepsis screening policy at ED triage", REQUIREMENTS)


def _metrics(d: pd.DataFrame, alert) -> dict:
    alert = np.asarray(alert, dtype=bool)
    y = d[TARGET].to_numpy()
    m = binary_metrics(y, alert)
    sens = {s: binary_metrics(y[(d.site == s).to_numpy()], alert[(d.site == s).to_numpy()])["recall"] for s in ["A", "B"]}
    return {"sensitivity %": m["recall"] * 100, "alerts per 100 visits": alert.mean() * 100,
            "missed per 1,000 visits": m["FN"] / len(y) * 1000, "PPV %": m["precision"] * 100,
            "site A sensitivity %": sens["A"] * 100, "site B sensitivity %": sens["B"] * 100,
            "site B / site A sensitivity": sens["B"] / sens["A"] if sens["A"] > 0 else 0.0}


def evaluate_policy(name: str, decide, kind: str, simplicity: float, why: str = "") -> dict:
    """Evaluate one screening policy on the VALIDATION month and add it to the scorecard.
    decide(d) -> boolean array of alerts; d has every triage column plus `risk` (the model's probability).
    kind: one of KINDS. simplicity: 1.0 = one threshold, 0.7 = two numbers / a combination, 0.4 = per-segment rules."""
    assert kind in KINDS, f"kind must be one of {sorted(KINDS)}"
    d = with_risk(val)
    res = _metrics(d, decide(d))
    shuffled = val.assign(**{LEAKY: val[LEAKY].sample(frac=1, random_state=1).to_numpy()})
    res["leak-free"] = bool(np.array_equal(np.asarray(decide(d), bool), np.asarray(decide(with_risk(shuffled)), bool)))
    res["simplicity"] = float(simplicity)
    alert, y = np.asarray(decide(d), bool), d[TARGET].to_numpy()
    rng, pos = np.random.default_rng(0), np.flatnonzero(y == 1)
    boot = [alert[rng.choice(pos, len(pos))].mean() * 100 for _ in range(1000)]     # resample the septic patients
    res["sensitivity 95% CI low"] = float(np.percentile(boot, 2.5))
    _state["policies"][name], _state["results"][name], _state["kinds"][name] = decide, res, kind
    card.add(name, res, note=why)
    show(f"evaluated: {name}", f"sens {res['sensitivity %']:.1f}% (CI low {res['sensitivity 95% CI low']:.1f}), "
                               f"{res['alerts per 100 visits']:.1f} alerts/100, "
                               f"site ratio {res['site B / site A sensitivity']:.2f}, leak-free {res['leak-free']}")
    return res


def site_report(name: str) -> pd.DataFrame:
    """Validation metrics of one policy, per site (where does it fail?)."""
    d = with_risk(val)
    alert = np.asarray(_state["policies"][name](d), bool)
    tab = segment_report(d[TARGET], alert.astype(float), 0.5, d.site.map({"A": "site A (city)", "B": "site B (rural)"}))
    _state["segments_done"].add(name)
    print(tab.round(3).to_string())
    return tab


def open_test_set() -> pd.DataFrame:
    """Apply EVERY registered policy, unchanged, to the sealed test month. Call ONCE."""
    _state["test_opened"] += 1
    d = with_risk(_test)
    tab = pd.DataFrame({n: _metrics(d, f(d)) for n, f in _state["policies"].items()}).T
    _state["test"] = tab
    print(tab.round(2).to_string())
    return tab


step("A2  How good is the risk model? (validation)")
dv = with_risk(val)
show("Validation ROC-AUC / average precision", f"{roc_auc_score(dv[TARGET], dv.risk):.3f} / "
                                              f"{average_precision_score(dv[TARGET], dv.risk):.3f}")
show("Lactate missing: site A / site B", f"{val[val.site == 'A'].lactate.isna().mean():.0%} / "
                                         f"{val[val.site == 'B'].lactate.isna().mean():.0%}")

step("A3  Baselines")
evaluate_policy("Baseline: qSOFA >= 2 (clinical rule)", lambda d: qsofa(d) >= 2, kind="other", simplicity=1.0,
                why="The bedside score clinicians use today")
evaluate_policy("Baseline: model, default threshold 0.5", lambda d: d.risk >= 0.5, kind="statistical", simplicity=1.0,
                why="What happens if nobody chooses a threshold")
note("""Neither baseline is acceptable: both miss far too many septic patients. Your
job is not a better model; it is a better POLICY for using the scores.""")

# %% =========================================================================
# PART B - YOUR WORK
# ============================================================================
banner("PART B - Threshold strategies, audit, decision (your work)")

step("B0  (optional) A different risk model")
# TODO (optional): MODEL = <your pipeline>.fit(train, train[TARGET])
#       Ask the leakage question for every column: "is this known at triage?"

step("B1  At least FOUR threshold strategies of at least THREE different kinds")
# Every threshold must be chosen using TRAIN or VALIDATION data only (never _test).
# Use with_risk(val) to get validation scores to tune on. Kinds: see KINDS.
# Example of the expected style (a fixed-sensitivity threshold):
# dv = with_risk(val)
# t90 = float(np.quantile(dv.risk[dv[TARGET] == 1], 0.10))            # catches 90% of validation cases
# evaluate_policy("S1 sensitivity >= 90%", lambda d: d.risk >= t90, kind="fixed-sensitivity", simplicity=1.0,
#                 why="Safety-first: fix the share of septic patients caught")
# Ideas: an alert budget (top 20% of risk), a cost-based threshold (COST_RATIO -> p* = 1 / (1 + COST_RATIO)),
#        a site-aware rule (a different threshold where lactate is missing), model OR qSOFA >= 2.
# Watch the 'CI low' printed for each policy: validation has only ~160 septic patients. A policy that
# passes the sensitivity gate with no margin may fail it next month.

step("B2  Site audit of your recommended policy")
# TODO: site_report("<name>") for at least your recommended policy. Which site does it fail, and why?

step("B3  Weight scenarios for the sensitivity analysis")
# TODO: at least TWO alternative weightings of the four soft criteria, in this order:
#       [Missed sepsis, Alert load, Positive predictive value, Simplicity]
SCENARIOS = {
    # "safety-first": [0.7, 0.1, 0.1, 0.1],
    # "capacity-first": [0.3, 0.5, 0.1, 0.1],
}

step("B4  Open the test set ONCE (after your choice is made on validation)")
# TODO: test_table = open_test_set()

step("B5  Justification")
# TODO: >= 150 words. Name the recommended policy, cite validation AND test numbers, state which hard gates
#       decided, what the site audit showed, what the sensitivity analysis showed, and the limitations
#       (e.g. flu season doubles the prevalence; a new hospital without lactate testing).
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

test_ok = None
if _state["test"] is not None and winner is not None:
    tw = _state["test"].loc[winner]
    test_ok = {r.name: bool(ModelScorecard._passes(r, tw.get(r.metric, _state["results"][winner].get(r.metric))))
               for r in REQUIREMENTS if r.hard}
    show("Winner's hard gates on the TEST month", test_ok)

candidates = [n for n in _state["results"] if not n.startswith("Baseline")]
words = JUSTIFICATION.split()
numbers = [w for w in words if any(ch.isdigit() for ch in w)]
checks = {
    ">= 4 strategies besides the baselines (6 pts)": len(candidates) >= 4,
    ">= 3 different kinds of strategy (6 pts)": len({_state["kinds"][n] for n in candidates}) >= 3,
    "No strategy uses post-triage information (6 pts)": bool(candidates) and all(
        _state["results"][n]["leak-free"] for n in candidates),
    "Requirements not loosened (4 pts)": ORIGINAL <= {(r.name, r.threshold, r.weight) for r in REQUIREMENTS},
    "Site audit done for the recommended policy (4 pts)": winner in _state["segments_done"],
    ">= 2 weight scenarios in the sensitivity analysis (4 pts)": len(SCENARIOS) >= 2,
    "Test set opened exactly once (4 pts)": _state["test_opened"] == 1,
    "Justification: >= 150 words, names the winner, cites >= 3 numbers (6 pts)": len(words) >= 150
    and winner is not None and winner in JUSTIFICATION and len(numbers) >= 3,
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
points = sum(int(k.split("(")[-1].split()[0]) for k, v in checks.items() if v)
show("Automated rubric points", f"{points} / 40")

if all(checks.values()):
    card.to_markdown(OUT_DIR / "assignment_sepsis_policy_memo.md", justification=JUSTIFICATION.strip(),
                     header=f"Train {len(train):,} visits (days 0-119); policies tuned on validation ({len(val):,} visits, "
                            f"days 120-149); test = last 30 days ({len(_test):,} visits, opened once). "
                            f"Sensitivity analysis: {sens}. Winner's gates on test: {test_ok}")
    card.to_json(OUT_DIR / "assignment_sepsis_policy.json")
    print("  [saved] outputs/assignment_sepsis_policy_memo.md + outputs/assignment_sepsis_policy.json")
else:
    note("""Not all checks pass yet - keep going with PART B. The memo is written
automatically once every check is PASS.""")
