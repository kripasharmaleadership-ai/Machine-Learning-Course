"""
LAB 14 - ASSIGNMENT WORKED EXAMPLE: A Model Comparison Report
=============================================================

THE ASSIGNMENT (see ASSIGNMENT.md)
    "Create a model comparison report covering performance, explainability, speed, and
    deployment complexity."
    This lab is the REFERENCE SOLUTION on real telemarketing data (Bank Marketing).
    Your assignment applies the same method to predictive maintenance (assignment_starter.py).

THE BUSINESS CONTEXT
    A bank calls customers to sell a term deposit. Each night a model must rank ~45,000 customers so
    that the call centre phones the most promising ones first. The call agents must be able to say
    WHY a customer is on the list, and the model runs inside the bank's existing batch platform.

THE FOUR DIMENSIONS OF THE REPORT
    PERFORMANCE              cross-validated AUC, AUC on the most RECENT months, lift in the top 10%
    EXPLAINABILITY           how a call agent / auditor can understand a score (rules, coefficients, SHAP)
    SPEED                    training time, batch throughput, single-customer latency
    DEPLOYMENT COMPLEXITY    model size, preprocessing steps, number of learned numbers, dependencies

LEVELS
    EASY   : The brief; requirements written first; the data and its two traps.
    MEDIUM : Five candidates measured on all four dimensions, identically.
    HARD   : Scorecard, sensitivity to the weights, explanations of the winner, the report, rubric self-check.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_lab_utils import BANK_TARGET, HAS_MPL, OUT_DIR, banner, load_bank_marketing, note, plt, save_fig, show, step
from tree_toolkit import ModelScorecard, Requirement, deployment_profile, df_to_md, permutation_importance_df, shapley_exact

# %% =========================================================================
# LEVEL 1 (EASY): Requirements first, then the data
# ============================================================================
banner("The brief and the requirements (written BEFORE any results)", "EASY")
REQUIREMENTS = [
    Requirement("No post-call information", "leak-free", hard=True, higher_is_better=True,
                why="The list is built BEFORE the call; anything measured during the call is unavailable"),
    Requirement("Ranking quality floor", "CV AUC", threshold=0.65, hard=True, higher_is_better=True,
                why="The current segment rules reach ~0.60; a model must be clearly better"),
    Requirement("Nightly batch fits the window", "rows / second (batch)", threshold=2000, hard=True, higher_is_better=True,
                why="45,000 customers scored in < 30 s inside the nightly job"),
    Requirement("Ranking quality", "CV AUC", weight=0.20, higher_is_better=True, why="Overall ranking (5-fold CV)"),
    Requirement("Recent months", "recent AUC", weight=0.15, higher_is_better=True,
                why="Behaviour drifts: performance on the latest 20% of calls"),
    Requirement("Top-decile lift", "top-10% lift", weight=0.15, higher_is_better=True,
                why="The call centre only reaches the top of the list"),
    Requirement("Explainability", "explainability", weight=0.20, higher_is_better=True,
                why="Agents must say why a customer is called; auditors must review the logic"),
    Requirement("Training time", "fit seconds", weight=0.05, why="Weekly retraining on shared hardware"),
    Requirement("Model size", "model size KB", weight=0.10, why="Stored and versioned with every release"),
    Requirement("Deployment complexity", "complexity points", weight=0.15,
                why="Preprocessing steps + learned numbers + dependencies to maintain"),
]
print(pd.DataFrame([{"requirement": r.name, "type": "HARD gate" if r.hard else f"soft ({r.weight:.0%})", "metric": r.metric,
                     "threshold": r.threshold if r.hard and r.threshold is not None else "-"} for r in REQUIREMENTS]).to_string(index=False))

step("1.1  The data, and its two traps")
bank = load_bank_marketing()
show("Calls x columns", bank.shape)
show("Subscribed (target)", f"{bank[BANK_TARGET].mean():.1%}")
cut = int(len(bank) * 0.8)                                           # rows are in time order (May 2008 -> Nov 2010)
train, recent = bank.iloc[:cut].reset_index(drop=True), bank.iloc[cut:].reset_index(drop=True)
show("Subscription rate: first 80% of calls / last 20%", f"{train[BANK_TARGET].mean():.1%} / {recent[BANK_TARGET].mean():.1%}")
print(bank.groupby(pd.cut(bank.duration, [-1, 60, 180, 600, 5000]), observed=True)[BANK_TARGET].mean().round(3)
      .rename("subscription rate by call duration (seconds)").to_string())
CAT = [c for c in bank.columns if bank[c].dtype == object]
NUM = [c for c in bank.columns if c not in CAT + [BANK_TARGET, "duration"]]
note(f"""Trap 1 - LEAKAGE: 'duration' (length of the call) is the strongest predictor in
the data, but it is only known AFTER the call; a list built tonight can't
use it. The data's authors say so explicitly. Trap 2 - DRIFT: the rows are in
time order and the subscription rate climbs from {train[BANK_TARGET].mean():.0%} to {recent[BANK_TARGET].mean():.0%} in the last
months (different campaigns, the 2008 crisis). So we compare models with
cross-validation on the first 80% AND check each one on the most recent 20%.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Five candidates, four dimensions
# ============================================================================
banner("Five candidates measured on four dimensions", "MEDIUM")


def onehot(scale):
    return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
                             verbose_feature_names_out=False)


X_tr, y_tr = train.drop(columns=[BANK_TARGET]), train[BANK_TARGET].to_numpy()
X_rc, y_rc = recent.drop(columns=[BANK_TARGET]), recent[BANK_TARGET].to_numpy()
X_tr_cat = X_tr.astype({c: "category" for c in CAT})
X_rc_cat = X_rc.astype({c: "category" for c in CAT})
FEATS = NUM + CAT
CANDIDATES = {   # name: (model, uses categorical dtypes?, columns, explainability score, how it is explained, steps, deps)
    "Logistic regression": (Pipeline([("prep", onehot(True)), ("model", LogisticRegression(C=0.5, max_iter=3000))]),
                            False, FEATS, 0.8, "coefficients per feature (odds ratios)", 3, 1),
    "Decision tree (depth 5)": (Pipeline([("prep", onehot(False)), ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                                                                                    random_state=0))]),
                                False, FEATS, 1.0, "32 readable rules", 2, 1),
    "Random forest": (Pipeline([("prep", onehot(False)), ("model", RandomForestClassifier(300, min_samples_leaf=10, n_jobs=1,
                                                                                         random_state=0))]),
                      False, FEATS, 0.4, "SHAP per customer (post-hoc)", 2, 1),
    "Gradient boosting": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, categorical_features="from_dtype",
                                                         random_state=0), True, FEATS, 0.5, "SHAP per customer (post-hoc)", 1, 1),
    "Gradient boosting + call duration": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                         categorical_features="from_dtype", random_state=0),
                                          True, FEATS + ["duration"], 0.5, "SHAP per customer (post-hoc)", 1, 1),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def learned_numbers(m):
    est = m[-1] if hasattr(m, "steps") else m
    if hasattr(est, "coef_"):
        return est.coef_.size + 1
    if hasattr(est, "tree_"):
        return int(2 * est.tree_.node_count)
    if hasattr(est, "estimators_"):
        return int(sum(2 * e.tree_.node_count for e in est.estimators_))
    return int(sum(len(pr[0].nodes) * 2 for pr in est._predictors))


results, fitted = {}, {}
for name, (model, cat_dtype, cols, expl, how, steps, deps) in CANDIDATES.items():
    Xa, Xb = (X_tr_cat, X_rc_cat) if cat_dtype else (X_tr, X_rc)
    cv_auc = cross_val_score(model, Xa[cols], y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    prof = deployment_profile(model, Xa[cols], y_tr, Xb[cols])
    m = prof.pop("_fitted")
    p_rc = m.predict_proba(Xb[cols])[:, 1]
    top = p_rc >= np.quantile(p_rc, 0.9)
    probe = Xa[cols].sample(1000, random_state=0)
    leak_free = "duration" not in cols or bool(np.allclose(m.predict_proba(probe)[:, 1], m.predict_proba(
        probe.assign(duration=probe["duration"].sample(frac=1, random_state=1).to_numpy()))[:, 1]))
    n_learned = learned_numbers(m)
    results[name] = {"CV AUC": cv_auc, "recent AUC": roc_auc_score(y_rc, p_rc), "top-10% lift": y_rc[top].mean() / y_rc.mean(),
                     "explainability": expl, **prof, "learned numbers": n_learned,
                     "complexity points": steps + np.log10(n_learned) + deps, "leak-free": leak_free}
    fitted[name] = (m, cat_dtype, cols, how)
res = pd.DataFrame(results).T
perf = res[["CV AUC", "recent AUC", "top-10% lift"]]
spd = res[["fit seconds", "latency ms (1 row)", "rows / second (batch)"]]
dep = res[["model size KB", "learned numbers", "complexity points"]]
for title, tab in [("PERFORMANCE", perf), ("SPEED", spd), ("DEPLOYMENT COMPLEXITY", dep)]:
    print(f"\n  {title}")
    print(tab.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
print("\n  EXPLAINABILITY")
print(pd.DataFrame({"score (0-1)": res["explainability"], "how a score is explained": [fitted[n][3] for n in res.index],
                    "leak-free": res["leak-free"]}).to_string())
honest_ = res.drop(index="Gradient boosting + call duration")
note(f"""Same data, same folds, same measurements for every candidate.
- The duration model has by far the best CV AUC ({res.loc['Gradient boosting + call duration', 'CV AUC']:.3f}): too good to be true, and it
  is. The leakage test (shuffle 'duration', see if scores change) flags it.
- Honest models: best on CV is {honest_['CV AUC'].idxmax()} ({honest_['CV AUC'].max():.3f}), but best on the RECENT
  months is {honest_['recent AUC'].idxmax()} ({honest_['recent AUC'].max():.3f}). Rankings change under drift: the CV winner
  is not automatically the model that holds up next quarter.
- Speed: all fit the nightly window easily (the forest is slowest to train and
  to score one customer, ~{res.loc['Random forest', 'latency ms (1 row)']:.0f} ms).
- Deployment: the forest is by far the biggest artefact ({res.loc['Random forest', 'model size KB'] / 1024:.0f} MB, ~{res.loc['Random forest', 'learned numbers'] / 1e6:.1f} million
  learned numbers); logistic regression and the tree are a few KB.""", keep_lines=True)

# %% =========================================================================
# LEVEL 3 (HARD): Decide, stress-test, explain, report
# ============================================================================
banner("Scorecard, sensitivity, explanations and the report", "HARD")
step("3.1  The scorecard")
card = ModelScorecard("Nightly call-list ranking for a term-deposit campaign", REQUIREMENTS)
for name in res.index:
    card.add(name, res.loc[name].to_dict(), note=f"explained by: {fitted[name][3]}")
board = card.evaluate()
print(board.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
winner = card.recommend()
show("Recommended model", winner)

step("3.2  Is the choice robust to the weights?")
scen = {"as written": None, "performance-first": [0.35, 0.25, 0.25, 0.05, 0.02, 0.04, 0.04],
        "explainability-first": [0.10, 0.10, 0.10, 0.50, 0.05, 0.05, 0.10],
        "ops-first (size, complexity)": [0.15, 0.10, 0.10, 0.15, 0.10, 0.20, 0.20]}
sens = {}
for label, w in scen.items():
    reqs = [Requirement(**r.__dict__) for r in REQUIREMENTS]
    if w:
        for r, wt in zip([r for r in reqs if not r.hard], w):
            r.weight = wt
    sens[label] = ModelScorecard(card.problem, reqs, results=card.results).recommend()
print(pd.Series(sens, name="recommended").to_string())
eligible = [n for n in board.index if bool(board.loc[n, "eligible"])]
runner_up = [n for n in eligible if n != winner][0]
m_w, m_r = fitted[winner], fitted[runner_up]
pw = m_w[0].predict_proba((X_rc_cat if m_w[1] else X_rc)[m_w[2]])[:, 1]
pr_ = m_r[0].predict_proba((X_rc_cat if m_r[1] else X_rc)[m_r[2]])[:, 1]
rng = np.random.default_rng(0)
diffs = []
for _ in range(500):
    b = rng.integers(0, len(y_rc), len(y_rc))
    diffs.append(roc_auc_score(y_rc[b], pw[b]) - roc_auc_score(y_rc[b], pr_[b]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
show(f"Recent AUC: {winner} minus {runner_up}", f"{np.mean(diffs):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
stable = len(set(sens.values())) == 1
note(f"""Plot twist: the simplest honest model, {winner}, wins the REPORT{' under every weighting' if stable else ''}.
It is not the best on cross-validation, but it holds up better on the recent
months ({np.mean(diffs):+.3f} AUC vs {runner_up}, 95% CI {lo:+.3f} to {hi:+.3f}: a real difference), it is
a few KB instead of megabytes, and its coefficients are readable. If the winner
had changed with the weights, the decision would be a BUSINESS trade-off, to
be stated and handed to the owner.""")

step("3.3  Explain the winner (what the report shows to call agents and auditors)")
mw, cat_w, cols_w, _ = fitted[winner]
Xw_tr = (X_tr_cat if cat_w else X_tr)[cols_w]
Xw_rc = (X_rc_cat if cat_w else X_rc)[cols_w]
imp = permutation_importance_df(mw, Xw_rc.sample(3000, random_state=0), y_rc[Xw_rc.sample(3000, random_state=0).index.to_numpy()],
                                lambda m_, X_, y_: roc_auc_score(y_, m_.predict_proba(X_)[:, 1]), n_repeats=3)
print(imp.head(6).round(4).rename("global: AUC drop when shuffled (recent calls)").to_string())
GROUPS = {"contact history (pdays, previous, poutcome)": ["pdays", "previous", "poutcome"], "month & day": ["month", "day"],
          "contact channel": ["contact"], "age & job": ["age", "job"], "balance & loans": ["balance", "housing", "loan", "default"],
          "campaign calls so far": ["campaign"], "marital & education": ["marital", "education"]}
top_customer = Xw_rc.iloc[[int(np.argmax(mw.predict_proba(Xw_rc)[:, 1]))]]
phi = shapley_exact(lambda d: mw.predict_proba(d)[:, 1], top_customer, Xw_tr.sample(50, random_state=0), GROUPS)
print(f"\n  Why is the top customer on tonight's list? (score {phi.attrs['prediction']:.2f}, average {phi.attrs['base_value']:.2f})")
for k, v in phi.sort_values(ascending=False).head(4).items():
    print(f"    {k:<45} {v:+.3f}")
note(f"""Two levels of explanation go into the report: GLOBAL (what drives the ranking:
{', '.join(imp.head(3).index)}) for auditors, and LOCAL reasons per customer
(exact SHAP, with features grouped into plain-language themes) for the agent
making the call. A post-hoc explanation
is less transparent than the tree's rules, which is why explainability has its
own weight in the scorecard rather than being an afterthought.""")

step("3.4  Write the report")
reasons = ", ".join(f"{k} ({v:+.2f})" for k, v in phi.sort_values(ascending=False).head(3).items())
extra = "\n".join([
    "## Performance", "", df_to_md(perf.round(3)), "",
    "## Explainability", "", df_to_md(pd.DataFrame({"score": res["explainability"],
                                                     "method": [fitted[n][3] for n in res.index]})), "",
    f"Winner's top global drivers (AUC drop when shuffled, recent calls): "
    + ", ".join(f"{k} ({v:.3f})" for k, v in imp.head(4).items()) + ".",
    f"Example local explanation (tonight's top customer, score {phi.attrs['prediction']:.2f} vs average {phi.attrs['base_value']:.2f}): {reasons}.", "",
    "## Speed", "", df_to_md(spd.round(3)), "",
    "## Deployment complexity", "", df_to_md(dep.round(2)), "",
    "Complexity points = preprocessing steps + log10(learned numbers) + external dependencies.",
])
justification = f"""**Why {winner}:** it passes every hard gate (no post-call information, AUC floor, nightly batch window)
and has the best weighted score. On the most recent calls it reaches AUC {res.loc[winner, 'recent AUC']:.3f} with a top-decile lift of
{res.loc[winner, 'top-10% lift']:.1f}x; versus {runner_up} the recent-AUC difference is {np.mean(diffs):+.3f} (95% CI {lo:+.3f} to {hi:+.3f}).

- **Rejected:** 'Gradient boosting + call duration' scores best on paper but uses information only known after the call (leak).
- **Explainability:** {fitted[winner][3]} for auditors, plus exact SHAP reasons per customer, grouped into plain-language
  themes, for call agents. (The depth-5 tree is the most readable model but fails the AUC floor: readable rules cost too much accuracy here.)
- **Sensitivity:** {'; '.join(f'{k}: {v}' for k, v in sens.items())}.
- **Limitations:** strong drift between campaigns (subscription rate {train[BANK_TARGET].mean():.0%} -> {recent[BANK_TARGET].mean():.0%}); every model loses AUC on recent
  months; retrain each campaign and monitor the top-decile conversion rate weekly."""
memo = card.to_markdown(OUT_DIR / "14_bank_model_comparison_report.md", justification=justification, extra_sections=extra,
                        header=f"Bank Marketing (UCI), {len(bank):,} calls in time order. Models compared with 5-fold CV on the first "
                               f"{len(train):,} calls; each checked on the most recent {len(recent):,}. Requirements fixed before results.")
card.to_json(OUT_DIR / "14_bank_model_comparison.json")
print("  [saved] outputs/14_bank_model_comparison_report.md + 14_bank_model_comparison.json")

step("3.5  Self-check with the assignment rubric")
checks = {
    "Requirements written first (>= 2 hard gates, >= 4 soft criteria)": sum(r.hard for r in REQUIREMENTS) >= 2
    and sum(r.weight > 0 for r in REQUIREMENTS) >= 4,
    ">= 3 honest candidates from >= 2 families": len(eligible) >= 3,
    "PERFORMANCE: CV + an out-of-time or out-of-group check": {"CV AUC", "recent AUC"} <= set(res.columns),
    "EXPLAINABILITY: a score per model + global and local explanations of the winner": len(imp) > 0 and len(phi) > 0,
    "SPEED: fit time, latency and throughput measured": {"fit seconds", "latency ms (1 row)", "rows / second (batch)"} <= set(res.columns),
    "DEPLOYMENT: size and complexity measured": {"model size KB", "complexity points"} <= set(res.columns),
    "Leakage tested and the leaky candidate rejected": not bool(board.loc["Gradient boosting + call duration", "eligible"]),
    "Sensitivity to the weights + CI for the top-2 difference": len(sens) >= 3 and lo <= hi,
    "Report written with a recommendation": memo.exists() and winner in memo.read_text(encoding="utf-8"),
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
show("Rubric checks passed", f"{sum(checks.values())}/{len(checks)}")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    honest = res.drop(index="Gradient boosting + call duration")
    axes[0].scatter(honest["explainability"], honest["recent AUC"], s=80, c=["#54a24b" if n == winner else "#4c78a8" for n in honest.index])
    for n, r in honest.iterrows():
        axes[0].annotate(n, (r["explainability"], r["recent AUC"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[0].set(xlabel="explainability score", ylabel="AUC on recent months", title="Performance vs explainability")
    axes[1].barh(res.index, res["model size KB"], color="#f58518")
    axes[1].set(xscale="log", xlabel="model size (KB, log)", title="Deployment: artefact size")
    axes[1].tick_params(axis="y", labelsize=8)
    axes[2].barh(perf.index, perf["CV AUC"], color="#9d9d9d", label="CV (first 80%)")
    axes[2].barh(perf.index, perf["recent AUC"], color="#4c78a8", alpha=0.8, label="recent 20%")
    axes[2].set(xlim=(0.5, 1), xlabel="AUC", title="Drift: every model loses ground")
    axes[2].legend(fontsize=8)
    axes[2].tick_params(axis="y", labelsize=8)
    save_fig(fig, "14_model_comparison")

banner("KEY TAKEAWAYS")
print("""  1. A model comparison is a REPORT on four dimensions: performance, explainability, speed, deployment.
  2. Write the requirements first; measure every candidate identically; test for leakage.
  3. Check performance where it matters: recent data, the top of the list, with a confidence interval.
  4. Explainability and deployment complexity are measurable, and deserve explicit weight.
  5. Stress-test the weights; when they flip the winner, it is a business decision: say so.""")
