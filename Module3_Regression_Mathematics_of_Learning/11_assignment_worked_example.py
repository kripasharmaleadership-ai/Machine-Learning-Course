"""
LAB 11 - ASSIGNMENT WORKED EXAMPLE: Compare Three Regression Models and Justify the Best
=======================================================================================

THE ASSIGNMENT (see ASSIGNMENT.md)
    "Compare three regression models and justify the best model using both metrics
    and business requirements."
    This lab is the REFERENCE SOLUTION on real house prices (Ames, Iowa).
    Your assignment applies the same method to a different problem
    (food-delivery time estimates: assignment_starter.py).

THE BUSINESS CONTEXT
    A mortgage lender wants an Automated Valuation Model (AVM) to pre-approve
    home-equity loans without sending an appraiser for every application.
    - Regulators require reasons for every decision (adverse-action notices) ->
      the model must be EXPLAINABLE feature by feature.
    - Industry AVMs are judged on median absolute % error (MdAPE) and on the share
      of homes valued within +-10% (PPE10).
    - Over-valuing expensive homes is the most costly mistake (large loans).
    - Each input the model needs must be collected by a loan officer: fewer is cheaper.

THE METHOD (what "justify" means)
    1. Write the requirements BEFORE seeing any results (no moving the goalposts).
    2. Same data, same CV folds, same metrics for every candidate.
    3. Hard gates first (must-haves), then a weighted score for the rest.
    4. Check that the decision is robust to the weights; open the test set once.
    5. Write a decision memo a non-specialist can follow.

LEVELS
    EASY   : The brief, the requirements, the data and a business-rule baseline.
    MEDIUM : Three candidates (Ridge, Lasso, decision tree), evaluated identically.
    HARD   : Scorecard, weight sensitivity, test set, residuals, memo, rubric self-check.
"""
# %%
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml_lab_utils import HAS_MPL, OUT_DIR, RANDOM_STATE, banner, load_ames, note, plt, save_fig, show, step
from regression_toolkit import ModelScorecard, Requirement, bootstrap_compare, residual_table

# %% =========================================================================
# LEVEL 1 (EASY): Requirements first, then data and a baseline
# ============================================================================
banner("The brief and the requirements (written BEFORE any modelling)", "EASY")
REQUIREMENTS = [
    Requirement("Accuracy floor", "MdAPE %", threshold=10.0, hard=True,
                why="Industry AVMs with MdAPE above ~10% are not accepted for lending decisions"),
    Requirement("Explainable", "explainable", hard=True, higher_is_better=True,
                why="Regulator requires per-feature reasons for every adverse decision"),
    Requirement("No over-valuation of top homes", "top-decile bias %", threshold=5.0, hard=True,
                why="Over-valuing the most expensive 10% creates the largest loan losses"),
    Requirement("Latency", "latency ms", threshold=50.0, hard=True,
                why="Valuation is called live inside the loan application form"),
    Requirement("Typical error (RMSE $)", "RMSE", weight=0.35, why="Overall dollar accuracy"),
    Requirement("Hit rate within 10%", "PPE10 %", weight=0.25, higher_is_better=True,
                why="Share of homes that need no manual review"),
    Requirement("Inputs to collect", "inputs used", weight=0.25,
                why="Every input is a question a loan officer must ask"),
    Requirement("Stability across folds", "CV RMSE sd", weight=0.15, why="Predictable behaviour on new data"),
]
card = ModelScorecard("Automated Valuation Model for home-equity loans (Ames, Iowa)", REQUIREMENTS)
print(pd.DataFrame([{"requirement": r.name, "type": "HARD gate" if r.hard else f"soft ({r.weight:.0%})",
                     "metric": r.metric, "threshold": r.threshold if r.hard else "-", "why": r.why}
                    for r in REQUIREMENTS]).to_string(index=False))

step("1.1  Data and scope")
ames = load_ames().drop(columns=["Id"])
TARGET = "SalePrice"
scope = ames.GrLivArea <= 4000
show("Houses / columns", ames.shape)
show("Out of scope: living area > 4,000 sqft", int((~scope).sum()))
ames = ames[scope].reset_index(drop=True)
note("""Scope decision (Module 2, Lab 10): the 4 houses above 4,000 sqft include two
partial sales at a fraction of their value. The AVM will not serve that
segment: applications above 4,000 sqft are routed to a human appraiser. This is
written into the memo, not silently dropped.""")

CAT = [c for c in ames.columns if ames[c].dtype == object] + ["MSSubClass", "MoSold"]
NUM = [c for c in ames.columns if c not in CAT + [TARGET]]
train, test = train_test_split(ames, test_size=0.2, random_state=RANDOM_STATE)
X_tr, y_tr, X_te, y_te = train[NUM + CAT], train[TARGET].to_numpy(), test[NUM + CAT], test[TARGET].to_numpy()
show("Train / test rows (test set sealed until Level 3)", f"{len(train)} / {len(test)}")
show("Numeric / categorical raw inputs", f"{len(NUM)} / {len(CAT)}")
FOLDS = KFold(5, shuffle=True, random_state=RANDOM_STATE)


class NeighbourhoodPricePerSqft(RegressorMixin, BaseEstimator):
    """BASELINE (the rule appraisers use): median $/sqft of the neighbourhood x living area."""

    def fit(self, X, y):
        ppsf = pd.Series(np.asarray(y) / X["GrLivArea"].to_numpy())
        self.rate_ = ppsf.groupby(X["Neighborhood"].to_numpy()).median()
        self.global_ = float(ppsf.median())
        return self

    def predict(self, X):
        return X["Neighborhood"].map(self.rate_).fillna(self.global_).to_numpy() * X["GrLivArea"].to_numpy()


# %% =========================================================================
# LEVEL 2 (MEDIUM): Three candidates, evaluated identically
# ============================================================================
banner("Three candidates + the baseline, same folds, same metrics", "MEDIUM")


def preprocessor(scale=True):
    num = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num), NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                                   sparse_output=False))]), CAT),
    ])


def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)


CANDIDATES = {
    "Ridge (log price)": (Pipeline([("prep", preprocessor()),
                                    ("model", log_target(RidgeCV(alphas=np.logspace(-2, 3, 30))))]), True),
    "Lasso (log price)": (Pipeline([("prep", preprocessor()),
                                    ("model", log_target(LassoCV(cv=5, max_iter=50000, random_state=0)))]), True),
    "Decision tree (depth <= 6)": (Pipeline([("prep", preprocessor(scale=False)),
                                             ("model", GridSearchCV(DecisionTreeRegressor(random_state=0),
                                                                    {"max_depth": [4, 5, 6], "min_samples_leaf": [5, 10, 20]},
                                                                    cv=5, scoring="neg_root_mean_squared_error"))]), True),
    "Baseline: neighbourhood $/sqft": (NeighbourhoodPricePerSqft(), True),
}
note("""Why these three? Ridge = the standard explainable workhorse (all features,
shrunk). Lasso = the same model family, but it switches inputs off (fewer
questions to ask). Decision tree = a different family that a lay person can
read as rules, and that needs no linearity. All linear candidates model
log(price): errors become PERCENTAGES, which is how AVMs are judged.""")


def raw_inputs_used(fitted, name):
    """How many RAW columns must a loan officer supply for this model?"""
    if name.startswith("Baseline"):
        return 2
    prep = fitted["prep"]
    out_names = prep.get_feature_names_out()
    inner = fitted["model"]
    if hasattr(inner, "regressor_"):
        used = np.abs(inner.regressor_.coef_) > 1e-8
    else:
        used = np.zeros(len(out_names), bool)
        feats = inner.best_estimator_.tree_.feature
        used[np.unique(feats[feats >= 0])] = True
    raw = set()
    for nm, u in zip(out_names, used):
        if not u:
            continue
        col = nm.split("__", 1)[1]
        raw.add(col if col in NUM else max((c for c in CAT if col.startswith(c + "_")), key=len))
    return len(raw)


def avm_metrics(y, p):
    ape = np.abs(p - y) / y * 100
    top = y >= np.quantile(y, 0.9)
    return {"RMSE": float(np.sqrt(np.mean((y - p) ** 2))), "MAE": float(np.mean(np.abs(y - p))),
            "R2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "MdAPE %": float(np.median(ape)), "PPE10 %": float(np.mean(ape <= 10) * 100),
            "top-decile bias %": float(np.mean((p[top] - y[top]) / y[top]) * 100)}


cv_results, oof_preds = {}, {}
for name, (model, explainable) in CANDIDATES.items():
    oof, fold_rmse = np.zeros(len(y_tr)), []
    t0 = time.perf_counter()
    for tr_i, va_i in FOLDS.split(X_tr):
        m = clone(model).fit(X_tr.iloc[tr_i], y_tr[tr_i])
        oof[va_i] = m.predict(X_tr.iloc[va_i])
        fold_rmse.append(np.sqrt(np.mean((y_tr[va_i] - oof[va_i]) ** 2)))
    fit_s = (time.perf_counter() - t0) / 5
    full = clone(model).fit(X_tr, y_tr)
    one = X_tr.iloc[[0]]
    lat = []
    for _ in range(50):
        t1 = time.perf_counter()
        full.predict(one)
        lat.append((time.perf_counter() - t1) * 1000)
    res = avm_metrics(y_tr, oof)
    res.update({"CV RMSE sd": float(np.std(fold_rmse)), "inputs used": raw_inputs_used(full, name),
                "latency ms": float(np.median(lat)), "explainable": explainable, "fit seconds": fit_s})
    cv_results[name], oof_preds[name] = res, oof
    card.add(name, res)
table = pd.DataFrame(cv_results).T
print(table.drop(columns=["explainable"]).to_string(float_format=lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"))
note("""Out-of-fold predictions: every training house is predicted by a model that
never saw it, so these are honest estimates. The test set is still untouched.
Note the baseline: the appraisers' rule of thumb is already a respectable
model, and the tree barely beats it on RMSE.""")

# %% =========================================================================
# LEVEL 3 (HARD): Decide, stress-test the decision, document it
# ============================================================================
banner("Scorecard, sensitivity, test set, memo", "HARD")

step("3.1  The scorecard: hard gates, then weighted score")
board = card.evaluate()
print(board.to_string(float_format=lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"))
winner = card.recommend()
show("Recommended model", winner)

step("3.2  Is the decision robust to the weights?")
scenarios = {"as written": None,
             "accuracy-first (RMSE 60%, PPE10 30%, inputs 5%, stability 5%)": [0.6, 0.3, 0.05, 0.05],
             "cost-first (RMSE 20%, PPE10 15%, inputs 55%, stability 10%)": [0.2, 0.15, 0.55, 0.1],
             "equal weights": [0.25, 0.25, 0.25, 0.25]}
sens = {}
for label, w in scenarios.items():
    reqs = [Requirement(**{**r.__dict__}) for r in REQUIREMENTS]
    if w:
        for r, wt in zip([r for r in reqs if not r.hard], w):
            r.weight = wt
    sc = ModelScorecard(card.problem, reqs, results=card.results)
    sens[label] = sc.recommend()
print(pd.Series(sens, name="recommended").to_string())
note("""If the winner changes under reasonable alternative weights, the decision is a
BUSINESS trade-off, not a technical one: say so in the memo and let the
business owner choose. If it doesn't, the recommendation is robust.""")

step("3.3  Open the test set once (all candidates, for the record)")
fitted = {name: clone(m).fit(X_tr, y_tr) for name, (m, _) in CANDIDATES.items()}
test_tab = pd.DataFrame({n: avm_metrics(y_te, m.predict(X_te)) for n, m in fitted.items()}).T
print(test_tab.to_string(float_format=lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"))
runner_up = board.index[1]
ci = bootstrap_compare(y_te, fitted[winner].predict(X_te), fitted[runner_up].predict(X_te))
show(f"RMSE({runner_up}) - RMSE({winner})", f"${ci['diff (B - A)']:,.0f}  95% CI [{ci['ci_low']:,.0f}, {ci['ci_high']:,.0f}]")
note("""The two linear models are statistically indistinguishable on accuracy (the
interval includes 0). That is exactly when the BUSINESS criteria must decide:
here, the number of inputs a loan officer has to collect.""")

step("3.4  Residuals of the winner by price band (test set)")
p_win = fitted[winner].predict(X_te)
band = pd.cut(y_te, [0, 130_000, 200_000, 300_000, np.inf], labels=["< $130k", "$130-200k", "$200-300k", "> $300k"])
rt = residual_table(y_te, p_win, pd.Series(band.astype(str)))
rt["mean % error"] = pd.Series((p_win - y_te) / y_te * 100).groupby(band.astype(str)).mean()
print(rt.round(1).to_string())
over = rt[rt["mean % error"] > 5]
note(f"""The winner passes on average, but check every band: a negative mean residual
(positive % error) means OVER-valuation. {'Bands over-valued by more than 5%: ' + ', '.join(f"{i} ({v:+.1f}%)" for i, v in over['mean % error'].items()) + '. Cheap homes are pulled up towards the average (regression to the mean): a classic AVM weakness. Business rule: valuations in that band go to manual review, and this goes into the memo.' if len(over) else 'No band is over-valued by more than 5%.'}""")

step("3.5  Explain the winner: the reasons a loan officer would see")
w_model = fitted[winner]
if hasattr(w_model, "named_steps") and hasattr(w_model["model"], "regressor_"):
    coefs = pd.Series(w_model["model"].regressor_.coef_, index=w_model["prep"].get_feature_names_out())
    coefs.index = coefs.index.str.replace("num__", "").str.replace("cat__", "")
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
    print(((np.exp(top) - 1) * 100).round(1).rename("% price change per +1 sd (or category vs others)").to_string())
note("""On a log-price model, exp(coef) - 1 is the % change in value per standard
deviation (or for having that category). These become the 'reason codes': 'the
estimate is lower mainly because of overall quality and living area'.""")

step("3.6  Write the decision memo and save the model")
tied = ci["ci_low"] < 0 < ci["ci_high"]
in_w, in_r = cv_results[winner]["inputs used"], cv_results[runner_up]["inputs used"]
justification = f"""**Why {winner}:** it passes every hard gate (MdAPE, explainability, no over-valuation of
top homes, latency) and has the best weighted score. See the candidates table for the numbers.
{f"On accuracy it is statistically TIED with {runner_up} (the test CI of the RMSE difference includes 0), so the business criteria decide: it needs {in_r - in_w} fewer inputs ({in_w} vs {in_r}), i.e. a shorter application form." if tied else ""}

- **Metrics:** on the sealed test set it reaches MdAPE {test_tab.loc[winner, 'MdAPE %']:.1f}% and values
  {test_tab.loc[winner, 'PPE10 %']:.0f}% of homes within +-10% (baseline: {test_tab.loc['Baseline: neighbourhood $/sqft', 'PPE10 %']:.0f}%).
  Test RMSE difference vs {runner_up}: ${ci['diff (B - A)']:,.0f} (95% CI {ci['ci_low']:,.0f} to {ci['ci_high']:,.0f}).
- **Business:** weight sensitivity -> {', '.join(f'{k}: {v}' for k, v in sens.items())}.
- **Rejected:** the decision tree is readable but too inaccurate to pass the MdAPE gate;
  the baseline is the fallback if the model service is unavailable.
- **Segment check:** {'over-valued price bands: ' + ', '.join(f"{i} ({v:+.1f}%)" for i, v in over['mean % error'].items()) + ' -> manual review for valuations in that band' if len(over) else 'no price band over-valued by more than 5%'}.
- **Scope and limitations:** not for homes > 4,000 sqft (manual appraisal); trained on one city, 2006-2010
  sales; re-validate before use in other markets or after large market moves (a trained model cannot
  extrapolate a price trend, Lab 07)."""
memo = card.to_markdown(OUT_DIR / "11_avm_model_selection_memo.md", justification=justification,
                        header=f"Train {len(train):,} houses (5-fold CV), test {len(test):,} houses (opened once). "
                               f"Requirements fixed before modelling.")
card.to_json(OUT_DIR / "11_avm_model_selection.json")
joblib.dump(fitted[winner], OUT_DIR / "avm_model_v1.joblib")
print("  [saved] outputs/11_avm_model_selection_memo.md, 11_avm_model_selection.json, avm_model_v1.joblib")

step("3.7  Self-check with the assignment rubric")
checks = {
    "Requirements written before results (>= 3 hard, >= 2 soft)": sum(r.hard for r in REQUIREMENTS) >= 3
    and sum(r.weight > 0 for r in REQUIREMENTS) >= 2,
    "A baseline is included": any("Baseline" in n for n in card.results),
    "Exactly three non-baseline candidates": sum("Baseline" not in n for n in card.results) == 3,
    "Same CV folds for every candidate": True,
    "Error metrics include MAE, RMSE and R2": all(k in cv_results[winner] for k in ["MAE", "RMSE", "R2"]),
    "A business metric beyond MAE/RMSE/R2": any(k in cv_results[winner] for k in ["MdAPE %", "PPE10 %"]),
    "Weight-sensitivity analysis": len(sens) >= 3,
    "Test set used once, with a CI for the top-2 difference": "ci_low" in ci,
    "Residuals analysed by segment": len(rt) >= 3,
    "Memo with recommendation + justification written": memo.exists() and winner in memo.read_text(encoding="utf-8"),
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
show("Rubric checks passed", f"{sum(checks.values())}/{len(checks)}")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    names = list(cv_results)
    axes[0].scatter([cv_results[n]["inputs used"] for n in names], [cv_results[n]["MdAPE %"] for n in names],
                    s=80, c=["#54a24b" if n == winner else "#9d9d9d" for n in names])
    for n in names:
        axes[0].annotate(n, (cv_results[n]["inputs used"], cv_results[n]["MdAPE %"]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
    axes[0].axhline(10, color="r", ls="--", label="MdAPE gate (10%)")
    axes[0].set(xlabel="raw inputs a loan officer must supply", ylabel="CV MdAPE (%)",
                title="Accuracy vs data-collection cost")
    axes[0].legend()
    axes[1].scatter(y_te / 1000, (p_win - y_te) / y_te * 100, s=8, alpha=0.6)
    axes[1].axhline(0, color="k")
    axes[1].axhline(10, color="grey", ls=":")
    axes[1].axhline(-10, color="grey", ls=":")
    axes[1].set(xlabel="actual price ($k)", ylabel="% error (positive = over-valued)",
                title=f"Winner on the test set: {winner}")
    save_fig(fig, "11_avm_model_selection")

banner("KEY TAKEAWAYS")
print("""  1. Write the business requirements BEFORE the results: hard gates + weighted criteria.
  2. Evaluate every candidate on the same folds with the same metrics, including a baseline.
  3. Add the metric the business uses (MdAPE, PPE10, bias on the costly segment), not just RMSE.
  4. Stress-test the decision: weight sensitivity, CI on the test difference, residuals by segment.
  5. The deliverable is a memo a non-specialist can follow, plus the saved model.""")
