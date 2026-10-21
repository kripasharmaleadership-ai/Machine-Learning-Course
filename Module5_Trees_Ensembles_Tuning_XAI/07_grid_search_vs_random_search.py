"""
LAB 07 - Hyperparameter Tuning: Grid Search vs Random Search
============================================================

THE LOOP
    for each candidate setting:  cross-validate it  ->  keep the best  ->  refit on all training data
    GridSearchCV          try EVERY combination of a list of values per hyperparameter
    RandomizedSearchCV    try n random combinations, drawn from distributions you specify
    HalvingRandomSearchCV try MANY candidates on a little data, keep the best third, give them
                          more data, repeat ('successive halving': spend compute on the promising ones)

WHY RANDOM OFTEN BEATS GRID (Bergstra & Bengio, 2012)
    Usually only one or two hyperparameters really matter. A 4 x 4 grid tries just 4 values of
    the important one (16 fits); 16 random draws try 16 different values of it, for the same cost.

THE CATCH: the best CV score of a search is OPTIMISTIC
    Picking the maximum of many noisy scores selects lucky noise (the 'winner's curse').
    Nested cross-validation (search inside each outer fold) gives an honest estimate of
    'tuning + model' as a whole.

LEVELS
    EASY   : A grid search, end to end; reading cv_results_.
    MEDIUM : Grid vs random at equal budget, repeated; tuning a gradient booster; successive halving.
    HARD   : The winner's curse, measured; nested CV; tuning on pure noise.
"""
# %%
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (enables HalvingRandomSearchCV)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.tree import DecisionTreeClassifier

from ml_lab_utils import CREDIT_NUM, HAS_MPL, RANDOM_STATE, banner, make_credit_applicants, note, plt, save_fig, show, step

credit = make_credit_applicants(n=10_000, seed=7)
X = pd.get_dummies(credit[CREDIT_NUM + ["employment"]], columns=["employment"], dtype=float)
y = credit["default"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# %% =========================================================================
# LEVEL 1 (EASY): A grid search, end to end
# ============================================================================
banner("GridSearchCV on a decision tree", "EASY")
grid = {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 20, 100, 300]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
res = pd.DataFrame(gs.cv_results_)
table = res.pivot_table(index="param_max_depth", columns="param_min_samples_leaf", values="mean_test_score",
                        dropna=False)
table.index = table.index.map(lambda v: "None" if pd.isna(v) else int(v))
print("  Mean CV ROC-AUC (rows = max_depth, columns = min_samples_leaf):")
print(table.round(3).to_string())
show("Best setting", gs.best_params_)
show("Best mean CV AUC / its sd across folds", f"{gs.best_score_:.4f} / {res.loc[gs.best_index_, 'std_test_score']:.4f}")
show("Fits performed", f"{len(res)} settings x 5 folds + 1 refit = {len(res) * 5 + 1}")
show("Test AUC of the refitted best model", f"{roc_auc_score(y_te, gs.predict_proba(X_te)[:, 1]):.4f}")
note("""16 settings x 5 folds = 80 fits, then the best setting is refitted on all the
training data (refit=True) and is what gs.predict uses. The table shows the
INTERACTION between the knobs: a big leaf size rescues an unlimited depth, and
a shallow depth makes leaf size irrelevant. That's why hyperparameters are
tuned together, not one at a time.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Grid vs random, and tuning a booster
# ============================================================================
banner("Grid vs random at the same budget", "MEDIUM")
step("2.1  One important knob (min_samples_leaf), one that barely matters (max_features in [0.7, 1])")


def tree_cv(leaf, mf):
    return cross_val_score(DecisionTreeClassifier(min_samples_leaf=int(leaf), max_features=mf, random_state=0),
                           X_tr, y_tr, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc").mean()


surface_leaf = np.unique(np.logspace(0, np.log10(800), 25).astype(int))
surface = {lf: tree_cv(lf, 1.0) for lf in surface_leaf}
true_best = max(surface.values())
grid_leaf, grid_mf = [1, 10, 100, 800], [0.7, 0.8, 0.9, 1.0]
grid_best = max(tree_cv(lf, mf) for lf in grid_leaf for mf in grid_mf)
random_best = []
for seed in range(12):
    r = np.random.default_rng(seed)
    leafs = np.exp(r.uniform(0, np.log(800), 16))                     # log-uniform over the same range
    mfs = r.uniform(0.7, 1.0, 16)
    random_best.append(max(tree_cv(lf, mf) for lf, mf in zip(leafs, mfs)))
rows = {"4 x 4 grid (16 fits, 4 distinct leaf sizes)": {"best CV AUC": grid_best, "gap to the best possible": true_best - grid_best},
        "random, 16 draws (mean over 12 searches)": {"best CV AUC": np.mean(random_best),
                                                     "gap to the best possible": true_best - np.mean(random_best)},
        "random, 16 draws (worst of 12 searches)": {"best CV AUC": np.min(random_best),
                                                    "gap to the best possible": true_best - np.min(random_best)}}
gvr = pd.DataFrame(rows).T
print(gvr.round(4).to_string())
show("Best possible (fine sweep of leaf size)", f"{true_best:.4f} at min_samples_leaf = {max(surface, key=surface.get)}")
note(f"""The grid spends 12 of its 16 fits re-testing the same 4 leaf sizes with
different max_features, which hardly matters; random search tries 16 different
leaf sizes for the same price. Honest result: this time the grid got LUCKY.
One of its 4 values ({min(grid_leaf, key=lambda g: abs(np.log(g) - np.log(max(surface, key=surface.get))))}) sits right next to the optimum ({max(surface, key=surface.get)}), and its best score
even beats the noise-free sweep, because 16 tries of an unimportant knob are 16
lottery tickets (the winner's curse, Level 3). Random search is insurance, not
magic: its WORST run out of 12 was within {gvr.iloc[2, 1]:.3f} of the best possible. The decisive
argument is cost: with 5 knobs, a grid of 4 values each is 4^5 = 1,024 settings
(5,120 fits); random search lets you pick the budget. And sample 'scale'
knobs (leaf size, learning rate, regularisation) on a LOG scale.""")

step("2.2  Tuning a gradient booster with RandomizedSearchCV")
dists = {"learning_rate": loguniform(0.01, 0.3), "max_leaf_nodes": randint(4, 64), "min_samples_leaf": randint(5, 200),
         "l2_regularization": loguniform(1e-3, 10), "max_features": uniform(0.4, 0.6)}
base = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, n_iter_no_change=20, random_state=0)
t0 = time.perf_counter()
rs = RandomizedSearchCV(base, dists, n_iter=25, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc",
                        random_state=0, n_jobs=-1).fit(X_tr, y_tr)
rs_s = time.perf_counter() - t0
t0 = time.perf_counter()
hs = HalvingRandomSearchCV(base, dists, n_candidates=80, factor=3, resource="n_samples", min_resources=400,
                           cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc", random_state=0,
                           n_jobs=-1).fit(X_tr, y_tr)
hs_s = time.perf_counter() - t0
default_cv = cross_val_score(base, X_tr, y_tr, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc").mean()
rows = {"defaults (no tuning)": {"candidates": 1, "seconds": np.nan, "best CV AUC": default_cv,
                                 "test AUC": roc_auc_score(y_te, base.fit(X_tr, y_tr).predict_proba(X_te)[:, 1])},
        "RandomizedSearchCV (25 candidates)": {"candidates": 25, "seconds": rs_s, "best CV AUC": rs.best_score_,
                                               "test AUC": roc_auc_score(y_te, rs.predict_proba(X_te)[:, 1])},
        "HalvingRandomSearchCV (80 candidates)": {"candidates": 80, "seconds": hs_s, "best CV AUC": hs.best_score_,
                                                  "test AUC": roc_auc_score(y_te, hs.predict_proba(X_te)[:, 1])}}
tune = pd.DataFrame(rows).T
tune["candidates"] = tune["candidates"].astype(int)
print(tune.to_string(float_format=lambda v: f"{v:.4f}"))
print("  Best random-search setting:", {k: round(float(v), 4) for k, v in rs.best_params_.items()})
note(f"""Honest result: tuning adds about {tune.iloc[1, 3] - tune.iloc[0, 3]:+.3f} test AUC over the defaults. Modern
boosters ship with good defaults; tuning matters more for trees and forests
(Labs 02 and 04) and on harder data. Successive halving screened 80 candidates
on small subsamples and spent full-size fits only on the best few: 3x more
candidates in a third of the time, and the same test AUC. (Its 'best CV AUC'
is measured in the last round on a subsample: compare searches on test.) Rule of thumb: tune the 3-4 knobs that control
capacity, with random or halving search, and stop when the gains are inside
the CV noise (Lab 06, 2.2).""")

# %% =========================================================================
# LEVEL 3 (HARD): The winner's curse
# ============================================================================
banner("The best CV score of a search is optimistic", "HARD")
step("3.1  Small data, big search: inner CV vs nested CV vs a large test set")
small_X, small_y = X_tr.iloc[:600], y_tr[:600]
search_space = {"max_depth": randint(1, 15), "min_samples_leaf": randint(1, 120), "max_features": uniform(0.2, 0.8),
                "criterion": ["gini", "entropy"]}
search = RandomizedSearchCV(DecisionTreeClassifier(random_state=0), search_space, n_iter=60, scoring="roc_auc",
                            cv=StratifiedKFold(5, shuffle=True, random_state=0), random_state=0)
search.fit(small_X, small_y)
nested = cross_val_score(search, small_X, small_y, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=1))
wc = pd.Series({"best inner CV score (what the search reports)": search.best_score_,
                "nested CV (search inside each outer fold)": nested.mean(),
                "reality: 3,000 test applicants": roc_auc_score(y_te, search.predict_proba(X_te)[:, 1])}, name="ROC-AUC")
print(wc.round(4).to_string())
note(f"""With 600 applicants and 60 candidate settings, the search reports {wc.iloc[0]:.3f}: the
best of 60 noisy scores, so partly luck. Nested CV re-runs the WHOLE search
inside each outer fold and scores the winner on data the search never saw:
{wc.iloc[1]:.3f}, much closer to the truth ({wc.iloc[2]:.3f}). Report nested CV (or a held-out test
set) for 'how good is my tuned model', and the inner score only for choosing.""")

step("3.2  Tuning on pure noise")
rng = np.random.default_rng(5)
noise_rows = {}
for n_rows in [600, 200]:                                            # same search, less data
    noise_y = rng.integers(0, 2, n_rows)
    noisy = RandomizedSearchCV(DecisionTreeClassifier(random_state=0), search_space, n_iter=150, scoring="roc_auc",
                               cv=StratifiedKFold(5, shuffle=True, random_state=0), random_state=0).fit(
        small_X.iloc[:n_rows], noise_y)
    scores = pd.Series(noisy.cv_results_["mean_test_score"])
    noise_rows[f"{n_rows} applicants, coin-flip labels"] = {"best CV AUC of 150 settings": noisy.best_score_,
                                                            "median setting": scores.median(),
                                                            "truth": 0.5}
noise_tab = pd.DataFrame(noise_rows).T
print(noise_tab.round(3).to_string())
note(f"""Nothing can be learned from coin flips (truth: AUC 0.5), yet the best of 150
settings beats the typical setting in cross-validation: {noise_tab.iloc[0, 0]:.3f} vs {noise_tab.iloc[0, 1]:.3f} with
600 applicants, and {noise_tab.iloc[1, 0]:.3f} vs {noise_tab.iloc[1, 1]:.3f} with 200. Every extra candidate is another
lottery ticket, and with less data each ticket is noisier, so the winning
score overstates the truth more. Huge searches on small data deserve the
most suspicion.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(list(surface), list(surface.values()), "k-", label="CV AUC vs leaf size")
    for lf in grid_leaf:
        axes[0].axvline(lf, color="#e45756", ls=":", lw=1)
    r = np.random.default_rng(0)
    for lf in np.exp(r.uniform(0, np.log(800), 16)):
        axes[0].axvline(lf, color="#4c78a8", ls="-", lw=0.6, alpha=0.6)
    axes[0].set(xscale="log", xlabel="min_samples_leaf (red: grid, blue: 16 random draws)", ylabel="CV AUC",
                title="Grid tests 4 values; random tests 16")
    axes[1].hist(random_best, bins=8, color="#4c78a8", alpha=0.7, label="random search (12 repeats)")
    axes[1].axvline(grid_best, color="#e45756", lw=2, label="4 x 4 grid")
    axes[1].axvline(true_best, color="k", ls="--", label="best possible")
    axes[1].set(xlabel="best CV AUC found with 16 fits", title="Same budget, better coverage")
    axes[1].legend(fontsize=8)
    axes[2].hist(scores, bins=25, color="#9d9d9d")
    axes[2].axvline(0.5, color="k", ls="--", label="truth (coin flips)")
    axes[2].axvline(noisy.best_score_, color="#e45756", label="best of 150")
    axes[2].set(xlabel="CV AUC of each setting", title="Tuning on pure noise")
    axes[2].legend(fontsize=8)
    save_fig(fig, "07_grid_vs_random")

banner("KEY TAKEAWAYS")
print("""  1. A search = cross-validate each candidate, keep the best, refit it on all training data.
  2. Tune interacting knobs together; sample scale knobs on a log scale.
  3. At equal budget, random search covers the important knob far better than a grid.
  4. Successive halving screens many candidates cheaply; good defaults often get you most of the way.
  5. The best CV score is optimistic (winner's curse): report nested CV or a held-out test set.""")
