"""
LAB 10 - Random Forests and an Introduction to Gradient Boosting
================================================================

TWO WAYS TO COMBINE MANY TREES (ensembles)
    RANDOM FOREST (bagging)                         GRADIENT BOOSTING
    - many DEEP trees, trained INDEPENDENTLY        - many SMALL trees, trained ONE AFTER ANOTHER
    - each on a bootstrap sample of the rows,       - each new tree fits the ERRORS the ensemble
      trying a random subset of features per split    still makes (the gradient of the log loss)
    - average their probabilities                   - add them up (scaled by a learning rate)
    - fixes VARIANCE (unstable trees, Lab 09)       - fixes BIAS (one small tree is too simple)
    - hard to over-fit by adding trees              - CAN over-fit: needs early stopping

GRADIENT BOOSTING FOR CLASSIFICATION, in 5 lines
    F_0 = log-odds of the base rate                   (start: everyone gets the average risk)
    repeat: p = sigmoid(F);  r = y - p                (r = negative gradient of log loss)
            fit a small REGRESSION tree to r          (WHERE is the model still wrong?)
            leaf value = sum(r) / sum(p (1 - p))      (HOW FAR to move: one Newton step per leaf)
            F = F + learning_rate * leaf value(x)     (nudge the scores in that direction)

LEVELS
    EASY   : Bagging by hand: 50 bootstrap trees averaged beat any single tree; out-of-bag score.
    MEDIUM : Random-forest knobs; gradient boosting written from scratch = scikit-learn;
             learning rate x number of trees, and early stopping.
    HARD   : The showdown: six classifiers, same folds, accuracy AND cost AND speed AND
             explainability, with a confidence interval for the winner's margin.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, banner, fraud_features, make_transactions, note, plt, save_fig, show,
                          step)
from classification_toolkit import bootstrap_compare, sigmoid

tx = make_transactions(n=80_000, seed=1)
X, y = fraud_features(tx), tx["is_fraud"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
show("Train / test transactions (frauds)", f"{len(y_tr):,} ({y_tr.sum()}) / {len(y_te):,} ({y_te.sum()})")

# %% =========================================================================
# LEVEL 1 (EASY): Bagging by hand
# ============================================================================
banner("Fifty unstable trees, averaged", "EASY")
rng = np.random.default_rng(0)
n_trees = 50
tree_probs = []
for b in range(n_trees):
    boot = rng.integers(0, len(X_tr), len(X_tr))                          # bootstrap: sample rows WITH replacement
    t = DecisionTreeClassifier(min_samples_leaf=5, max_features="sqrt", random_state=b).fit(X_tr.iloc[boot], y_tr[boot])
    tree_probs.append(t.predict_proba(X_te)[:, 1])
tree_probs = np.array(tree_probs)
single_ap = [average_precision_score(y_te, p) for p in tree_probs]
rows = {"one tree (best of 50)": max(single_ap), "one tree (average of 50)": np.mean(single_ap)}
for k in [5, 20, 50]:
    rows[f"average of {k} trees"] = average_precision_score(y_te, tree_probs[:k].mean(axis=0))
rf50 = RandomForestClassifier(50, min_samples_leaf=5, oob_score=True, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
rows["RandomForestClassifier(50)"] = average_precision_score(y_te, rf50.predict_proba(X_te)[:, 1])
print(pd.Series(rows, name="test average precision").round(3).to_string())
show("Out-of-bag accuracy (free validation)", f"{rf50.oob_score_:.4f}")
note("""Each tree is trained on a bootstrap sample (the same number of rows, drawn with
replacement: ~63% unique) and may only consider a random subset of features at
each split. Every tree is noisy, but their errors are partly independent, so
averaging cancels much of the noise: the forest beats even the luckiest single
tree. The ~37% of rows a tree never saw ('out-of-bag') give a free validation
score. (OOB ACCURACY is printed here only because scikit-learn reports it: with 1.6%
fraud it is ~98% for any model. Use AP.)""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Forest knobs, boosting from scratch
# ============================================================================
banner("Random-forest knobs, and gradient boosting written by hand", "MEDIUM")
step("2.1  More trees never hurt a forest; max_features decorrelates them")
rows = {}
for n in [10, 50, 200]:
    for mf in ["sqrt", 1.0]:
        m = RandomForestClassifier(n, min_samples_leaf=5, max_features=mf, n_jobs=-1, random_state=0)
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        rows[(n, "sqrt (4 of 14)" if mf == "sqrt" else "all 14")] = {
            "test AP": average_precision_score(y_te, m.predict_proba(X_te)[:, 1]), "fit seconds": time.perf_counter() - t0}
rf_tab = pd.DataFrame(rows).T
rf_tab.index.names = ["trees", "features tried per split"]
print(rf_tab.round(3).to_string())
note("""Adding trees improves AP with diminishing returns, and never makes it worse:
an average of more trees is just a more precise average (it only costs time).
Trying a random subset of features at each split makes the trees DIFFERENT
from each other, which is what makes averaging work. With all features
('bagged trees') the trees are more alike.""")

step("2.2  Gradient boosting from scratch (log loss, regression-tree steps)")


def boost(X_a, y_a, X_b, n_rounds=150, lr=0.1, depth=3, newton=True):
    """Gradient boosting for binary classification, the textbook algorithm (Friedman, 2001)."""
    F_a = np.full(len(y_a), np.log(y_a.mean() / (1 - y_a.mean())))     # F_0 = base-rate log-odds
    F_b = np.full(len(X_b), F_a[0])
    history = []
    for _ in range(n_rounds):
        p = sigmoid(F_a)
        residual = y_a - p                                              # negative gradient of log loss
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_a, residual)
        leaf_a, leaf_b = t.apply(X_a), t.apply(X_b)
        if newton:                                                      # leaf value = sum(r) / sum(p(1-p))
            num = np.bincount(leaf_a, weights=residual, minlength=t.tree_.node_count)
            den = np.bincount(leaf_a, weights=p * (1 - p), minlength=t.tree_.node_count)
            value = num / np.maximum(den, 1e-12)
        else:                                                           # leaf value = mean residual
            value = t.tree_.value[:, 0, 0]
        F_a += lr * value[leaf_a]
        F_b += lr * value[leaf_b]
        history.append(log_loss(y_a, sigmoid(F_a)))
    return sigmoid(F_b), history


X_sub, y_sub = X_tr.iloc[:30_000], y_tr[:30_000]                     # a 30k-row subsample keeps the demo quick
p_ours, hist = boost(X_sub.to_numpy(), y_sub, X_te.to_numpy())
p_plain_steps, _ = boost(X_sub.to_numpy(), y_sub, X_te.to_numpy(), newton=False)
sk_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=0).fit(X_sub, y_sub)
p_sk = sk_gb.predict_proba(X_te)[:, 1]
rows = {"ours, leaf = mean residual (plain gradient step)": p_plain_steps, "ours, leaf = Newton step": p_ours,
        "sklearn GradientBoostingClassifier": p_sk}
print(pd.DataFrame({k: {"test ROC-AUC": roc_auc_score(y_te, v), "test AP": average_precision_score(y_te, v),
                        "test log loss": log_loss(y_te, v)} for k, v in rows.items()}).T.round(4).to_string())
show("Correlation: ours (Newton) vs sklearn", f"{np.corrcoef(p_ours, p_sk)[0, 1]:.4f}")
show("Training log loss: round 1 -> 50 -> 150", f"{hist[0]:.4f} -> {hist[49]:.4f} -> {hist[-1]:.4f}")
note("""Our booster starts from the base-rate log-odds, then repeatedly fits a small
REGRESSION tree to 'y - p' (where the model is still wrong) and adds a small
step to the scores. The tree decides WHERE to move; the leaf value decides HOW
FAR. With the plain mean residual the steps are tiny for a rare class (p ~ 0.02,
so y - p barely moves the log-odds) and 150 rounds are not enough. Dividing by
sum(p(1 - p)), one Newton step (curvature-aware), is what scikit-learn, XGBoost
and LightGBM do, and our 20 lines then reproduce scikit-learn. This is gradient
descent again (Module 3), but each step is a TREE instead of a weight update.""")

step("2.3  Learning rate x number of trees, and early stopping")
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=1)
curves, rows = {}, {}
for lr_ in [1.0, 0.3, 0.1, 0.03]:
    m = HistGradientBoostingClassifier(learning_rate=lr_, max_iter=600, max_leaf_nodes=15, early_stopping=False,
                                       random_state=0).fit(X_fit, y_fit)
    val_ll = [log_loss(y_val, p[:, 1]) for p in m.staged_predict_proba(X_val)]
    curves[lr_] = val_ll
    rows[f"learning_rate {lr_}"] = {"best # trees (validation)": int(np.argmin(val_ll)) + 1,
                                    "best val log loss": min(val_ll), "val log loss at 600 trees": val_ll[-1]}
print(pd.DataFrame(rows).T.astype({"best # trees (validation)": int}).round(4).to_string())
base_ll = log_loss(y_val, np.full(len(y_val), y_fit.mean()))
show("Validation log loss of 'predict the base rate'", f"{base_ll:.4f}")
es = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=2000, max_leaf_nodes=15, early_stopping=True,
                                    validation_fraction=0.2, n_iter_no_change=30, scoring="loss", random_state=0).fit(X_tr, y_tr)
show("Early stopping (lr 0.1) stopped after", f"{es.n_iter_} trees")
note("""Unlike a forest, boosting CAN over-fit: after the best number of trees, each
new tree chases noise and the validation loss rises again. With rare positives
a big learning rate is brutal: at 1.0 and 0.3 even the FIRST Newton steps
overshoot (leaves with a handful of frauds get huge values), and the model
never gets below the 'predict the base rate' loss. At 0.1 the best point comes
after a few dozen trees; at 0.03 it takes more trees but over-fits gently.
In practice: a modest learning rate + early stopping on a validation split
(one line in HistGradientBoostingClassifier).""")

# %% =========================================================================
# LEVEL 3 (HARD): The showdown
# ============================================================================
banner("Six classifiers, one fraud problem, same folds", "HARD")
interaction = FunctionTransformer(lambda d: d.assign(night_x_foreign=d.is_night * d.is_foreign))
CANDIDATES = {
    "Logistic regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), 1.0),
    "Logistic + night x foreign": (make_pipeline(interaction, StandardScaler(), LogisticRegression(max_iter=3000)), 1.0),
    "KNN (k=50)": (make_pipeline(StandardScaler(), KNeighborsClassifier(50)), 0.5),
    "Decision tree (depth 6)": (DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=0), 0.75),
    "Random forest (200)": (RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=0), 0.25),
    "Gradient boosting (HGB)": (HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                                               early_stopping=True, n_iter_no_change=30,
                                                               random_state=0), 0.25),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
results, oof_all = {}, {}
for name, (model, explain) in CANDIDATES.items():
    oof = np.zeros(len(y_tr))
    t0 = time.perf_counter()
    for a, b in cv.split(X_tr, y_tr):
        oof[b] = clone(model).fit(X_tr.iloc[a], y_tr[a]).predict_proba(X_tr.iloc[b])[:, 1]
    fit_s = (time.perf_counter() - t0) / 5
    full = clone(model).fit(X_tr, y_tr)
    lat = []
    for _ in range(20):
        t1 = time.perf_counter()
        full.predict_proba(X_te.iloc[[0]])
        lat.append((time.perf_counter() - t1) * 1000)
    f_, t_, _ = roc_curve(y_tr, oof)
    results[name] = {"CV ROC-AUC": roc_auc_score(y_tr, oof), "CV AP": average_precision_score(y_tr, oof),
                     "recall @ FPR 1%": float(np.interp(0.01, f_, t_)), "Brier": brier_score_loss(y_tr, oof),
                     "fit s / fold": fit_s, "ms / prediction": float(np.median(lat)), "explainability (0-1)": explain}
    oof_all[name] = (oof, full)
board = pd.DataFrame(results).T.sort_values("CV AP", ascending=False)
print(board.to_string(float_format=lambda v: f"{v:.3f}" if v < 100 else f"{v:,.0f}"))
first, second = board.index[0], board.index[1]
p1, p2 = oof_all[first][1].predict_proba(X_te)[:, 1], oof_all[second][1].predict_proba(X_te)[:, 1]
ci = bootstrap_compare(y_te, p1, p2, metric=average_precision_score, n_boot=1000)
show(f"Test AP: {first} / {second}", f"{average_precision_score(y_te, p1):.3f} / {average_precision_score(y_te, p2):.3f}")
show("AP(2nd) - AP(1st) on test, 95% CI", f"{ci['diff (B - A)']:+.3f}  [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]")
lr_plain = results["Logistic regression"]["CV AP"]
gb_ap = results["Gradient boosting (HGB)"]["CV AP"]
tied = ci["ci_low"] < 0 < ci["ci_high"]
note(f"""Same folds, same metrics, six families. Plot twist: the SIMPLEST family wins.
- Plain logistic regression ({lr_plain:.3f} AP) beats boosting ({gb_ap:.3f}) and the forest.
  The truth here is (almost) linear in log-odds, and with ~900 training frauds
  the flexible models pay for their flexibility in variance. Fancy is not
  automatically better: always keep the simple baseline in the race.
- The interaction the tree discovered in Lab 09 is worth something: ONE
  hand-made feature (night x foreign) gives the best model of all.
- Top two: '{first}' vs '{second}'. The test confidence interval of their
  difference {'includes 0: a statistical TIE' if tied else 'excludes 0: a real difference'} (~{y_te.sum()} test frauds).
- So the decision moves to the other columns: explainability (a regulator
  asking 'why was this card blocked?'), calibration (Brier) and latency. The
  forest's ~{results['Random forest (200)']['ms / prediction']:.0f} ms is thread start-up for ONE row (n_jobs=-1): for
  online scoring set n_jobs=1. Accuracy picks the shortlist; the business picks
  the model (Module 3, Lab 11).""", keep_lines=True)

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    sizes = [1, 2, 5, 10, 20, 30, 50]
    axes[0].plot(sizes, [average_precision_score(y_te, tree_probs[:k].mean(0)) for k in sizes], "o-", label="average of k trees")
    axes[0].axhline(np.mean(single_ap), color="grey", ls="--", label="a single tree (mean)")
    axes[0].set(xlabel="trees averaged", ylabel="test AP", title="Bagging: averaging unstable trees")
    axes[0].legend()
    for lr_, c in curves.items():
        axes[1].plot(range(1, len(c) + 1), c, label=f"learning rate {lr_}")
    axes[1].set(xlabel="boosting rounds (trees)", ylabel="validation log loss", ylim=(min(min(c) for c in curves.values()) * 0.97, 0.08),
                title="Boosting over-fits without early stopping")
    axes[1].legend()
    b = board.sort_values("CV AP")
    axes[2].barh(b.index, b["CV AP"], color=["#54a24b" if i in (first, second) else "#9d9d9d" for i in b.index])
    axes[2].set(xlabel="CV average precision", title="The showdown (5-fold CV)")
    axes[2].tick_params(axis="y", labelsize=8)
    save_fig(fig, "10_forest_boosting_showdown")

banner("KEY TAKEAWAYS")
print("""  1. Random forest = many deep trees on bootstrap samples with random features, averaged: less variance.
  2. Gradient boosting = small trees added one by one, each fitting the remaining error: less bias.
  3. Boosting is gradient descent where each step is a tree; tune learning rate + early stopping.
  4. A forest improves (slowly) with more trees; boosting over-fits after the best round.
  5. Compare families on the same folds; when the top models tie, explainability and cost decide.""")
