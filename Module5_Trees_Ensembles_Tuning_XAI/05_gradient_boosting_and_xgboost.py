"""
LAB 05 - Gradient Boosting and XGBoost
======================================

GRADIENT BOOSTING (Friedman, 2001) - recap from Module 4, Lab 10
    F_0 = a constant; then for m = 1..M: fit a small tree to the negative gradient of the loss,
    add it with a learning rate:  F_m = F_{m-1} + eta * tree_m

XGBOOST (Chen & Guestrin, 2016): the same idea, three upgrades
    1. SECOND-ORDER: each row contributes a gradient g_i and a hessian h_i (curvature).
       For log loss: g = p - y,  h = p (1 - p).  For squared error: g = prediction - y, h = 1.
    2. A REGULARISED objective:  loss + gamma x (number of leaves) + 1/2 lambda x sum(leaf value^2)
       optimal leaf value      w* = -G / (H + lambda)          (G, H = sums over the leaf's rows)
       gain of a split         1/2 [GL^2/(HL+l) + GR^2/(HR+l) - G^2/(H+l)] - gamma
    3. ENGINEERING: histograms of feature values, sparsity-aware splits (a learned default
       direction for missing values), row and column subsampling, parallel tree building.
    LightGBM (leaf-wise growth, GOSS) and CatBoost (ordered target statistics for categories)
    are variations on the same objective; scikit-learn's HistGradientBoosting is LightGBM-style.

LEVELS
    EASY   : One XGBoost split, by hand: gradients, hessians, leaf values, gain; what lambda and gamma do.
    MEDIUM : Our XGBoostLite vs scikit-learn's boosters; the regularisation knobs; early stopping.
    HARD   : Informative missing values and the learned default direction; monotonic constraints.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from ml_lab_utils import (CREDIT_NUM, HAS_MPL, RANDOM_STATE, banner, make_credit_applicants, note, optional_import, plt,
                          save_fig, show, step)
from tree_toolkit import XGBoostLite, xgb_gain, xgb_leaf_weight

# %% =========================================================================
# LEVEL 1 (EASY): One split, by hand
# ============================================================================
banner("XGBoost's split formula on 8 rows", "EASY")
node = pd.DataFrame({"late_payments": [0, 0, 0, 1, 1, 2, 3, 4], "defaulted": [0, 0, 1, 0, 0, 1, 1, 1]})
p0 = node.defaulted.mean()
node["p (current model)"] = p0                                       # first round: everyone gets the base rate
node["g = p - y"] = node["p (current model)"] - node.defaulted
node["h = p (1 - p)"] = p0 * (1 - p0)
print(node.to_string(index=False))
G, H = node["g = p - y"].sum(), node["h = p (1 - p)"].sum()
show("Node totals G / H", f"{G:.3f} / {H:.3f}")
rows = []
for cut in [0.5, 1.5, 2.5]:
    L = node.late_payments <= cut
    GL, HL = node.loc[L, "g = p - y"].sum(), node.loc[L, "h = p (1 - p)"].sum()
    GR, HR = G - GL, H - HL
    for lam, gam in [(0, 0), (1, 0), (1, 0.5)]:
        rows.append({"split": f"late <= {cut}", "lambda": lam, "gamma": gam,
                     "left leaf w = -GL/(HL+l)": xgb_leaf_weight(GL, HL, lam),
                     "right leaf w": xgb_leaf_weight(GR, HR, lam), "gain": xgb_gain(GL, HL, GR, HR, lam, gam)})
split_tab = pd.DataFrame(rows)
print(split_tab.round(3).to_string(index=False))
note("""Every row carries a gradient (which way the log-odds should move: p - y) and a
hessian (how confident that direction is: p(1 - p)). A leaf's best value is
-G/(H + lambda): the average push, shrunk by lambda. The gain compares the two
children with the parent; gamma is a fixed price per split, so weak splits
(gain < gamma) are simply not made: pruning built into the objective. The
best cut here is 'late <= 1.5', and lambda shrinks the leaf values towards
zero, more for small leaves than big ones.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): A working booster, and its knobs
# ============================================================================
banner("XGBoostLite vs scikit-learn, and the regularisation knobs", "MEDIUM")
credit = make_credit_applicants(n=20_000, seed=6, with_truth=True)
X = pd.get_dummies(credit[CREDIT_NUM + ["employment"]], columns=["employment"], dtype=float)
y = credit["default"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=1)

step("2.1  Three implementations of the same idea")
cands = {
    "XGBoostLite (ours, ~150 lines)": XGBoostLite(n_estimators=300, learning_rate=0.05, max_depth=3, reg_lambda=1.0),
    "sklearn GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                                     random_state=0),
    "sklearn HistGradientBoostingClassifier": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
                                                                             l2_regularization=1.0, early_stopping=False,
                                                                             random_state=0),
}
xgb = optional_import("xgboost")
if xgb is not None:
    cands["xgboost.XGBClassifier (installed)"] = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                                     reg_lambda=1.0, tree_method="hist")
rows = {}
for name, m in cands.items():
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    pr = m.predict_proba(X_te)[:, 1]
    rows[name] = {"test AUC": roc_auc_score(y_te, pr), "test log loss": log_loss(y_te, pr), "fit seconds": time.perf_counter() - t0}
impl = pd.DataFrame(rows).T
print(impl.round(4).to_string())
if xgb is None:
    print("  (xgboost is not installed: `pip install xgboost` adds it to this table automatically)")
note("""Same objective, same accuracy. Our booster is the textbook algorithm in pure
numpy: binned features, gradients and hessians, the gain formula, leaf values.
The libraries add speed (C++, multi-threading, cache-friendly histograms),
not a different idea. In practice use HistGradientBoosting (built into
scikit-learn), XGBoost, LightGBM or CatBoost; they differ mostly in speed,
defaults and categorical handling, rarely by more than noise in accuracy.""")

step("2.2  The regularisation knobs (validation log loss after 300 rounds, lr 0.1, depth 5)")
base = dict(n_estimators=300, learning_rate=0.1, max_depth=5)
knobs = {"no regularisation (lambda 0)": dict(reg_lambda=0.0), "lambda = 1 (default)": dict(reg_lambda=1.0),
         "lambda = 20": dict(reg_lambda=20.0), "gamma = 2": dict(reg_lambda=1.0, gamma=2.0),
         "min_child_weight = 10": dict(reg_lambda=1.0, min_child_weight=10.0),
         "subsample 0.7 + colsample 0.7": dict(reg_lambda=1.0, subsample=0.7, colsample_bytree=0.7),
         "all of the above": dict(reg_lambda=20.0, gamma=2.0, min_child_weight=10.0, subsample=0.7, colsample_bytree=0.7)}
rows, curves = {}, {}
for name, kw in knobs.items():
    m = XGBoostLite(**base, **kw).fit(X_fit, y_fit, eval_set=(X_val, y_val))
    curves[name] = m.eval_history_
    rows[name] = {"total leaves": m.n_leaves(), "best val log loss": min(m.eval_history_),
                  "best round": int(np.argmin(m.eval_history_)) + 1, "val log loss at 300": m.eval_history_[-1]}
reg_tab = pd.DataFrame(rows).T
print(reg_tab.to_string(float_format=lambda v: f"{v:.4f}" if v < 10 else f"{v:,.0f}"))
note(f"""Deep trees (depth 5) and a fast learning rate over-fit quickly: without
regularisation the validation loss is best after only {reg_tab.iloc[0, 2]:.0f} rounds and then climbs.
lambda shrinks small leaves, gamma refuses weak splits (look at the number of
leaves), min_child_weight forbids leaves with too little evidence, and row /
column subsampling decorrelates the rounds (the random-forest trick inside
boosting). Combined, they push the best round later and make over-shooting
it far less costly.""")

step("2.3  Early stopping: let the validation set choose the number of trees")
rows = {}
for lr in [0.3, 0.1, 0.03]:
    m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=3000, max_leaf_nodes=15, early_stopping=True,
                                       validation_fraction=0.2, n_iter_no_change=30, random_state=0).fit(X_tr, y_tr)
    pr = m.predict_proba(X_te)[:, 1]
    rows[f"learning rate {lr}"] = {"trees kept": m.n_iter_, "test AUC": roc_auc_score(y_te, pr), "test log loss": log_loss(y_te, pr)}
print(pd.DataFrame(rows).T.round(4).to_string())
note("""Smaller learning rate = more, smaller steps = more trees, usually a slightly
better optimum. Early stopping makes 'how many trees?' a non-question: train
with a generous maximum and stop when the validation loss hasn't improved for
n_iter_no_change rounds. Tune the learning rate and tree size; let early
stopping pick the count.""")

# %% =========================================================================
# LEVEL 3 (HARD): Missing values and monotonic constraints
# ============================================================================
banner("Missing values that carry information; rules the model must respect", "HARD")

step("3.1  Informative missingness and the learned default direction")
rng = np.random.default_rng(3)
miss_p = np.where(y == 1, 0.35, 0.08)                            # defaulters skip the income question more often
X_miss = X.copy()
X_miss.loc[rng.random(len(X)) < miss_p, "income_k"] = np.nan
Xm_tr, Xm_te = X_miss.loc[X_tr.index], X_miss.loc[X_te.index]
show("Income missing: defaulters / non-defaulters", f"{np.isnan(X_miss.income_k[y == 1]).mean():.0%} / "
                                                    f"{np.isnan(X_miss.income_k[y == 0]).mean():.0%}")
rows = {}
variants = {
    "LOGISTIC: median imputation": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                 LogisticRegression(max_iter=3000)),
    "LOGISTIC: median imputation + missing flag": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                                StandardScaler(), LogisticRegression(max_iter=3000)),
    "median imputation, then boosting": make_pipeline(SimpleImputer(strategy="median"),
                                                      HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)),
    "median imputation + missing flag": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                      HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)),
    "native NaN handling (HistGradientBoosting)": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
    "native NaN handling (XGBoostLite)": XGBoostLite(n_estimators=300, learning_rate=0.05, max_depth=4),
}
for name, m in variants.items():
    rows[name] = {"test AUC": roc_auc_score(y_te, m.fit(Xm_tr, y_tr).predict_proba(Xm_te)[:, 1])}
miss_tab = pd.DataFrame(rows).T
print(miss_tab.round(4).to_string())
lite = variants["native NaN handling (XGBoostLite)"]
income_col = list(X.columns).index("income_k")
dirs = [nd.get("missing_left") for t in lite.trees_[:50] for nd in [t] if nd.get("feature") == income_col]
lr_gain = miss_tab.iloc[1, 0] - miss_tab.iloc[0, 0]
note(f"""Here MISSING income is itself a warning sign. For logistic regression, median
imputation erases that signal (a missing income becomes an ordinary income);
the missing flag gives it back: {lr_gain:+.3f} AUC. Plot twist for trees: the
boosters lose nothing with plain median imputation, because every imputed row
has EXACTLY the median, and a tree can carve out that one value: the constant
acts as a hidden flag. Native NaN handling does it openly: at every split the
booster tries sending the NaN rows left AND right and keeps the better side,
the learned default direction. Rule: linear models need an explicit flag;
trees find it, but native handling is cleaner and survives a change of
imputation value. (Missingness is a feature: Module 2, Lab 02.)""")

step("3.2  Monotonic constraints: more debt must never LOWER the risk")
probe = X_te.sample(1500, random_state=0)


def violations(model, feature, delta):
    """Share of applicants whose predicted risk goes DOWN when `feature` goes up by delta."""
    p0_ = model.predict_proba(probe)[:, 1]
    p1_ = model.predict_proba(probe.assign(**{feature: probe[feature] + delta}))[:, 1]
    return float(np.mean(p1_ < p0_ - 1e-9))


mono = np.zeros(X.shape[1], dtype=int)
for f_, sign in {"debt_ratio": 1, "late_payments": 1, "utilisation": 1, "credit_years": -1}.items():
    mono[list(X.columns).index(f_)] = sign
rows = {}
for name, m in {"unconstrained": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0),
                "monotonic_cst on 4 features": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                                              monotonic_cst=mono, random_state=0)}.items():
    m.fit(X_tr, y_tr)
    rows[name] = {"test AUC": roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]),
                  "risk falls when debt_ratio +0.05": violations(m, "debt_ratio", 0.05),
                  "risk falls with +1 late payment": violations(m, "late_payments", 1),
                  "risk falls when utilisation +0.05": violations(m, "utilisation", 0.05)}
mono_tab = pd.DataFrame(rows).T
print(mono_tab.round(4).to_string())
note(f"""An unconstrained booster learns some wiggles from noise: for
{mono_tab.iloc[0, 1]:.0%} of applicants, a HIGHER debt ratio gives a LOWER predicted risk. A regulator
(or a customer told 'pay down your debt to improve your score') can't accept
that. monotonic_cst forces the direction for chosen features; violations drop
to zero at (almost) no cost in AUC. Domain knowledge expressed as a
constraint is regularisation that also buys trust.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for name in ["no regularisation (lambda 0)", "lambda = 20", "subsample 0.7 + colsample 0.7", "all of the above"]:
        axes[0].plot(range(1, len(curves[name]) + 1), curves[name], label=name)
    axes[0].set(xlabel="boosting round", ylabel="validation log loss", title="Regularisation delays over-fitting",
                ylim=(min(min(c) for c in curves.values()) * 0.98, min(min(c) for c in curves.values()) * 1.25))
    axes[0].legend(fontsize=8)
    grid = np.linspace(0, 0.95, 60)
    one = X_te.iloc[[3]]
    for name, m in [("unconstrained", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                                     random_state=0).fit(X_tr, y_tr)),
                    ("monotonic", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                                 monotonic_cst=mono, random_state=0).fit(X_tr, y_tr))]:
        axes[1].plot(grid, m.predict_proba(pd.concat([one] * len(grid)).assign(debt_ratio=grid))[:, 1], label=name)
    axes[1].set(xlabel="debt ratio (one applicant, everything else fixed)", ylabel="P(default)",
                title="Monotonic constraint: no wiggles")
    axes[1].legend()
    save_fig(fig, "05_boosting_xgboost")

banner("KEY TAKEAWAYS")
print("""  1. XGBoost = gradient boosting + second-order steps + a regularised objective + engineering.
  2. Leaf value -G/(H + lambda); split gain minus gamma: regularisation is inside the formula.
  3. lambda, gamma, min_child_weight and subsampling delay over-fitting; early stopping picks the rounds.
  4. Boosters learn where missing values should go; missingness can carry signal.
  5. Monotonic constraints encode domain rules: no wiggles, almost no accuracy cost.""")
