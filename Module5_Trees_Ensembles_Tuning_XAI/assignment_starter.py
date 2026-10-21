"""
ASSIGNMENT STARTER - A model comparison report for predictive maintenance
=========================================================================

Read ASSIGNMENT.md first. Reference solution for a DIFFERENT problem (bank telemarketing):
14_assignment_worked_example_model_report.py

    python assignment_starter.py

The script runs as-is: PART A (data, requirements, evaluation harness, a baseline) and PART C
(scorecard, rubric, report) are written for you. Your work is PART B: build at least three
candidate models, explain the winner, stress-test the weights, open the test set ONCE, and
write the four report sections plus the recommendation. Re-run until every check is PASS.
"""
# %%
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ml_lab_utils import DATA_DIR, MAINT_TARGET, OUT_DIR, banner, make_machine_telemetry, note, show, step
from tree_toolkit import ModelScorecard, Requirement, df_to_md, permutation_importance_df, shapley_exact

RAW_PATH = DATA_DIR / "pump_telemetry.csv"
if not RAW_PATH.exists():                       # generated once, then treated as an immutable raw file
    make_machine_telemetry().to_csv(RAW_PATH, index=False)
tele = pd.read_csv(RAW_PATH)
TARGET, LEAKY = MAINT_TARGET, "downtime_hours_next_7d"

# %% =========================================================================
# PART A - Brief, data, requirements, harness (done for you)
# ============================================================================
banner("PART A - The brief")
print("""  A water utility runs 400 pumps in 4 plants. Every Monday a model scores each pump's latest weekly
  telemetry and the maintenance planner inspects the riskiest 10% that week. An unplanned failure costs
  ~$18,000 (emergency repair + downtime); an inspection costs ~$400. The model will run on the plant's
  EDGE GATEWAY (limited memory, no internet), and the planner must be able to explain to a plant manager
  why a pump was (or wasn't) on the list. Models will be used on NEW pumps too (the fleet grows).""")

machines = tele.machine_id.unique()
rng = np.random.default_rng(2026)
test_machines = rng.choice(machines, 80, replace=False)                # sealed: 80 pumps the model never sees
train = tele[~tele.machine_id.isin(test_machines)].reset_index(drop=True)
_test = tele[tele.machine_id.isin(test_machines)].reset_index(drop=True)
y_tr = train[TARGET].to_numpy()
show("Training rows / pumps", f"{len(train):,} / {train.machine_id.nunique()}")
show("Test rows / pumps (unseen pumps, sealed)", f"{len(_test):,} / {_test.machine_id.nunique()}")
show("Failure rate (next 7 days)", f"{y_tr.mean():.1%}")
print(train.head(3).T.to_string())

# The requirements are fixed by the operations team. You may ADD requirements, but not loosen these.
REQUIREMENTS = [
    Requirement("No future information", "leak-free", hard=True, higher_is_better=True,
                why="Scores are produced on Monday; nothing measured later in the week may be used"),
    Requirement("Works on new pumps", "grouped CV AUC", threshold=0.65, hard=True, higher_is_better=True,
                why="Cross-validated by PUMP: the fleet grows and new pumps must be scored well"),
    Requirement("Fits on the edge gateway", "model size KB", threshold=20_000, hard=True,
                why="The gateway has ~24 MB free for the model artefact"),
    Requirement("Ranking quality", "grouped CV AUC", weight=0.20, higher_is_better=True, why="Overall ranking of pumps"),
    Requirement("Failures caught in the top 10%", "recall @ top 10%", weight=0.20, higher_is_better=True,
                why="The planner inspects only the riskiest 10% each week"),
    Requirement("Explainability", "explainability", weight=0.20, higher_is_better=True,
                why="The planner must explain the list to plant managers"),
    Requirement("Training time", "fit seconds", weight=0.05, why="Retrained weekly on a small server"),
    Requirement("Latency", "latency ms (1 row)", weight=0.10, why="On-demand re-scoring from the gateway's UI"),
    Requirement("Model size", "model size KB", weight=0.10, why="Smaller artefacts update faster over the plant's link"),
    Requirement("Deployment complexity", "complexity points", weight=0.15,
                why="Preprocessing steps + learned numbers + dependencies to maintain on the gateway"),
]
ORIGINAL = {(r.name, r.threshold, r.weight) for r in REQUIREMENTS}
card = ModelScorecard("Weekly pump-inspection list (predictive maintenance)", REQUIREMENTS)
FOLDS = GroupKFold(5)
_state = {"models": {}, "results": {}, "families": {}, "explained": set(), "test_opened": 0, "test": None}


def _learned_numbers(m):
    est = m[-1] if hasattr(m, "steps") else m
    est = getattr(est, "best_estimator_", est)
    if hasattr(est, "coef_"):
        return int(np.size(est.coef_) + 1)
    if hasattr(est, "tree_"):
        return int(2 * est.tree_.node_count)
    if hasattr(est, "estimators_"):
        return int(sum(2 * e.tree_.node_count for e in np.ravel(est.estimators_)))
    if hasattr(est, "_predictors"):
        return int(sum(len(pr[0].nodes) * 2 for pr in est._predictors))
    return 1


def evaluate(name: str, model, explainability: float, how_explained: str, preprocessing_steps: int = 1,
             dependencies: int = 1) -> dict:
    """Measure one candidate on all four dimensions. `model` takes the RAW training DataFrame (use a Pipeline).
    explainability: 1.0 = readable rules, 0.8 = coefficients, 0.5 = post-hoc SHAP on a boosted model, 0.4 = forest + SHAP."""
    oof = np.full(len(train), np.nan)
    for a, b in FOLDS.split(train, y_tr, groups=train.machine_id):
        oof[b] = clone(model).fit(train.iloc[a], y_tr[a]).predict_proba(train.iloc[b])[:, 1]
    top = oof >= np.quantile(oof, 0.9)
    t0 = time.perf_counter()
    full = clone(model).fit(train, y_tr)
    fit_s = time.perf_counter() - t0
    lat = []
    for i in range(30):
        t1 = time.perf_counter()
        full.predict_proba(train.iloc[[i]])
        lat.append((time.perf_counter() - t1) * 1000)
    probe = train.sample(1000, random_state=0)
    shuffled = probe.assign(**{LEAKY: probe[LEAKY].sample(frac=1, random_state=1).to_numpy()})
    n_learned = _learned_numbers(full)
    res = {"grouped CV AUC": roc_auc_score(y_tr, oof), "recall @ top 10%": float(y_tr[top].sum() / y_tr.sum()),
           "explainability": float(explainability), "fit seconds": fit_s, "latency ms (1 row)": float(np.median(lat)),
           "model size KB": len(pickle.dumps(full)) / 1024, "learned numbers": n_learned,
           "complexity points": preprocessing_steps + np.log10(max(n_learned, 1)) + dependencies,
           "leak-free": bool(np.allclose(full.predict_proba(probe)[:, 1], full.predict_proba(shuffled)[:, 1]))}
    card.add(name, res, note=f"explained by: {how_explained}")
    est = full[-1] if hasattr(full, "steps") else full
    _state["models"][name], _state["results"][name] = full, res
    _state["families"][name] = type(getattr(est, "best_estimator_", est)).__name__
    show(f"evaluated: {name}", f"grouped CV AUC {res['grouped CV AUC']:.3f}, recall@10% {res['recall @ top 10%']:.0%}, "
                               f"{res['model size KB']:,.0f} KB, leak-free {res['leak-free']}")
    return res


def global_importance(name: str, n_repeats: int = 3) -> pd.Series:
    """Permutation importance of one fitted candidate on a held-out sample of training PUMPS."""
    m = _state["models"][name]
    sample = train[train.machine_id.isin(rng.choice(train.machine_id.unique(), 60, replace=False))]
    cols = [c for c in sample.columns if c not in (TARGET, "machine_id")]
    imp = permutation_importance_df(m, sample, sample[TARGET].to_numpy(),
                                    lambda m_, X_, y_: roc_auc_score(y_, m_.predict_proba(X_)[:, 1]), n_repeats=n_repeats)
    _state["explained"].add((name, "global"))
    imp = imp.drop(labels=[TARGET, "machine_id"], errors="ignore")
    print(imp.round(4).rename(f"{name}: AUC drop when shuffled").to_string())
    return imp


def local_explanation(name: str, row_index: int, features: list | dict) -> pd.Series:
    """Exact SHAP values for one training row (features: a list of columns or a dict {group name: [columns]})."""
    m = _state["models"][name]
    phi = shapley_exact(lambda d: m.predict_proba(d)[:, 1], train.iloc[[row_index]], train.sample(50, random_state=0), features)
    _state["explained"].add((name, "local"))
    print(f"  {name}, row {row_index}: P(failure) {phi.attrs['prediction']:.3f} vs average {phi.attrs['base_value']:.3f}")
    print(phi.sort_values(key=abs, ascending=False).round(4).to_string())
    return phi


def open_test_set() -> pd.DataFrame:
    """Score every candidate on the 80 sealed pumps. Call ONCE, after your choice is made."""
    _state["test_opened"] += 1
    rows = {}
    for name, m in _state["models"].items():
        p = m.predict_proba(_test)[:, 1]
        top = p >= np.quantile(p, 0.9)
        rows[name] = {"test AUC": roc_auc_score(_test[TARGET], p),
                      "test recall @ top 10%": float(_test[TARGET].to_numpy()[top].sum() / _test[TARGET].sum())}
    tab = pd.DataFrame(rows).T
    _state["test"] = tab
    print(tab.round(3).to_string())
    return tab


class ServiceIntervalRule(ClassifierMixin, BaseEstimator):
    """BASELINE used today: inspect the pumps that have gone longest without a service."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = np.clip(X["days_since_service"].to_numpy() / 300, 0, 1)
        return np.c_[1 - s, s]


step("A1  Baseline")
evaluate("Baseline: days since service", ServiceIntervalRule(), explainability=1.0,
         how_explained="one rule: longest since service first", preprocessing_steps=0, dependencies=0)

# %% =========================================================================
# PART B - YOUR WORK
# ============================================================================
banner("PART B - Candidates, explanations, decision, report (your work)")

step("B1  At least THREE candidates from at least TWO families")
# TODO: build three candidates (e.g. a pruned decision tree, a random forest, a gradient booster; logistic regression
#       is welcome too). Each is a scikit-learn Pipeline that takes the RAW `train` DataFrame and picks its own
#       columns (ColumnTransformer). Ask for every column: "is this known on Monday morning?"
# Example of the expected style (uncomment, then build your own):
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.tree import DecisionTreeClassifier
# NUM = ["install_age_years", "days_since_service", "temperature_c", "vibration_mm_s", "pressure_bar", "rpm", "load_factor"]
# CAT = ["plant", "pump_type"]
# prep = ColumnTransformer([("num", "passthrough", NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
# tree = Pipeline([("prep", prep), ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=0))])
# evaluate("M1 decision tree (depth 4)", tree, explainability=1.0, how_explained="16 readable rules", preprocessing_steps=2)

step("B2  Explain your recommended model: globally AND locally")
# TODO: global_importance("<name>") and local_explanation("<name>", row_index, features)
#       Tip: pick an interesting row (a pump the model ranks very high), and group related columns, e.g.
#       {"service": ["days_since_service"], "vibration & temperature": ["vibration_mm_s", "temperature_c"], ...}

step("B3  Weight scenarios for the sensitivity analysis")
# TODO: at least TWO alternative weightings of the 7 soft criteria, in this order:
#       [ranking, recall@10%, explainability, training time, latency, model size, complexity]
SCENARIOS = {
    # "maintenance-first": [0.30, 0.40, 0.10, 0.02, 0.03, 0.05, 0.10],
    # "plant-IT-first":    [0.10, 0.15, 0.15, 0.05, 0.15, 0.20, 0.20],
}

step("B4  Open the test set ONCE")
# TODO: test_table = open_test_set()

step("B5  The report: four sections and a recommendation")
# TODO: write each section in your own words, citing your numbers (>= 40 words each; recommendation >= 120 words
#       and it must name the recommended model). Mention the traps you found, and what you would monitor.
REPORT = {
    "PERFORMANCE": """
""",
    "EXPLAINABILITY": """
""",
    "SPEED": """
""",
    "DEPLOYMENT COMPLEXITY": """
""",
    "RECOMMENDATION": """
""",
}

# %% =========================================================================
# PART C - Scorecard, rubric and the report (done for you)
# ============================================================================
banner("PART C - Decide, self-check, write the report")
board = card.evaluate()
print(board.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
winner = card.recommend()
show("Recommended by the scorecard", winner)
sens = {"as written": winner}
for label, w in SCENARIOS.items():
    reqs = [Requirement(**r.__dict__) for r in REQUIREMENTS]
    for r, wt in zip([r for r in reqs if not r.hard], w):
        r.weight = wt
    sens[label] = ModelScorecard(card.problem, reqs, results=card.results).recommend()
print(pd.Series(sens, name="recommended").to_string())

candidates = [n for n in _state["results"] if not n.startswith("Baseline")]
words = {k: v.split() for k, v in REPORT.items()}
rec = REPORT["RECOMMENDATION"]
checks = {
    ">= 3 candidates besides the baseline (5 pts)": len(candidates) >= 3,
    ">= 2 model families (5 pts)": len({_state["families"][n] for n in candidates}) >= 2,
    "No candidate uses future information (6 pts)": bool(candidates) and all(_state["results"][n]["leak-free"] for n in candidates),
    "Requirements not loosened (4 pts)": ORIGINAL <= {(r.name, r.threshold, r.weight) for r in REQUIREMENTS},
    "Recommended model explained globally AND locally (6 pts)": {(winner, "global"), (winner, "local")} <= _state["explained"],
    ">= 2 weight scenarios (4 pts)": len(SCENARIOS) >= 2,
    "Test set opened exactly once (4 pts)": _state["test_opened"] == 1,
    "Report: 4 sections >= 40 words, recommendation >= 120 words naming the winner with >= 3 numbers (6 pts)":
        all(len(words[k]) >= 40 for k in ["PERFORMANCE", "EXPLAINABILITY", "SPEED", "DEPLOYMENT COMPLEXITY"])
        and len(words["RECOMMENDATION"]) >= 120 and winner is not None and winner in rec
        and sum(any(ch.isdigit() for ch in w) for w in words["RECOMMENDATION"]) >= 3,
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
points = sum(int(k.split("(")[-1].split()[0]) for k, v in checks.items() if v)
show("Automated rubric points", f"{points} / 40")

if all(checks.values()):
    res = pd.DataFrame(_state["results"]).T
    sections = "\n".join(
        [f"## {k.title()}\n\n{REPORT[k].strip()}\n" for k in ["PERFORMANCE", "EXPLAINABILITY", "SPEED", "DEPLOYMENT COMPLEXITY"]]
        + ["## Measurements\n", df_to_md(res.drop(columns=["leak-free"]).astype(float).round(3)), "",
           "## Test set (80 unseen pumps, opened once)\n", df_to_md(_state["test"].round(3)), ""])
    card.to_markdown(OUT_DIR / "assignment_pump_model_report.md", justification=rec.strip(), extra_sections=sections,
                     header=f"{train.machine_id.nunique()} pumps for training (5-fold GroupKFold by pump), 80 unseen pumps for "
                            f"testing. Sensitivity analysis: {sens}")
    card.to_json(OUT_DIR / "assignment_pump_model_report.json")
    print("  [saved] outputs/assignment_pump_model_report.md + outputs/assignment_pump_model_report.json")
else:
    note("""Not all checks pass yet - keep going with PART B. The report is written
automatically once every check is PASS.""")
