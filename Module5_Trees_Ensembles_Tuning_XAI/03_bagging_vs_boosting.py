"""
LAB 03 - Bagging vs Boosting
============================

TWO OPPOSITE WAYS TO BUILD A STRONG MODEL FROM MANY TREES
                      BAGGING (bootstrap aggregating)          BOOSTING
    base learner      DEEP trees (low bias, high variance)    SMALL trees (high bias, low variance)
    how trained       in PARALLEL, each on a bootstrap sample  SEQUENTIALLY, each fixes the errors so far
    how combined      AVERAGE                                  weighted SUM
    what it reduces   VARIANCE                                 BIAS
    risk              can't fix a biased learner               can over-fit if you add too many rounds

THE DECOMPOSITION WE WILL MEASURE (Module 3, Lab 04, now for probabilities)
    expected Brier score = bias^2 + variance + noise
        bias^2   : how far the AVERAGE prediction (over many training sets) is from the true probability
        variance : how much the prediction changes from one training set to another
        noise    : p(1 - p), what even a perfect model can't remove
    Only possible because the synthetic data gives us the TRUE default probability.

LEVELS
    EASY   : One picture, one dimension: a jagged deep tree vs its bagged average; a stump vs boosted stumps.
    MEDIUM : bias^2 and variance measured over 20 independent training sets for 8 models.
    HARD   : Why bagging needs UNSTABLE learners; the tree-correlation formula behind random forests;
             what happens to bias and variance as you add more trees to each.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml_lab_utils import CREDIT_NUM, HAS_MPL, banner, make_credit_applicants, note, plt, save_fig, show, step
from tree_toolkit import bias_variance_proba

# %% =========================================================================
# LEVEL 1 (EASY): The idea in one dimension
# ============================================================================
banner("Averaging vs adding, on a 1-D curve", "EASY")
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 10, 150))
f = lambda v: np.sin(v) + 0.3 * v                                     # the true curve
y1 = f(x) + rng.normal(0, 0.5, len(x))
grid = np.linspace(0, 10, 400)[:, None]
fits = {
    "one deep tree": DecisionTreeRegressor(random_state=0).fit(x[:, None], y1),
    "bagging: 200 deep trees, averaged": BaggingRegressor(DecisionTreeRegressor(), n_estimators=200, random_state=0).fit(x[:, None], y1),
    "one stump (1 question)": DecisionTreeRegressor(max_depth=1).fit(x[:, None], y1),
    "boosting: 200 stumps, added": GradientBoostingRegressor(max_depth=1, n_estimators=200, learning_rate=0.1,
                                                             random_state=0).fit(x[:, None], y1),
}
rows = {name: {"RMSE vs the TRUE curve": np.sqrt(np.mean((m.predict(grid) - f(grid[:, 0])) ** 2)),
               "train RMSE": np.sqrt(np.mean((m.predict(x[:, None]) - y1) ** 2))} for name, m in fits.items()}
one_d = pd.DataFrame(rows).T
print(one_d.round(3).to_string())
note("""A deep tree passes through every noisy point (train RMSE 0): low bias, high
variance. Averaging 200 deep trees fitted on bootstrap samples smooths the
noise away: same flexibility, much less variance. A stump is the opposite: one
step can't follow a curve (high bias). Adding 200 stumps, each fitting what is
still missing, builds up the curve step by step: boosting removes bias.
(outputs/03_bagging_vs_boosting.png shows the four fits.)""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Measure bias^2 and variance
# ============================================================================
banner("bias^2 and variance, measured over 20 training sets", "MEDIUM")
FEATS = CREDIT_NUM


def design(df):
    return pd.get_dummies(df[FEATS + ["employment"]], columns=["employment"], dtype=float).reindex(
        columns=FEATS + [f"employment_{e}" for e in ["contract", "salaried", "self_employed", "unemployed"]], fill_value=0)


test = make_credit_applicants(n=3000, seed=999, with_truth=True)
X_test, p_test = design(test), test["_true_prob"].to_numpy()
N_TRAIN = 1500


def sample_train(r):
    d = make_credit_applicants(n=N_TRAIN, seed=1000 + r)
    return design(d), d["default"].to_numpy()


MODELS = {
    "stump (depth 1)": lambda: DecisionTreeClassifier(max_depth=1, random_state=0),
    "tree depth 4": lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
    "deep tree (leaf >= 1)": lambda: DecisionTreeClassifier(random_state=0),
    "BAGGING 100 deep trees": lambda: BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, n_jobs=-1, random_state=0),
    "RANDOM FOREST 100 trees": lambda: RandomForestClassifier(100, n_jobs=-1, random_state=0),
    "RANDOM FOREST, leaf >= 10": lambda: RandomForestClassifier(100, min_samples_leaf=10, n_jobs=-1, random_state=0),
    "BOOSTING 200 stumps": lambda: GradientBoostingClassifier(max_depth=1, n_estimators=200, learning_rate=0.1, random_state=0),
    "BOOSTING 200 depth-3 trees": lambda: GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.1, random_state=0),
}
rows = {}
for name, mk in MODELS.items():
    bv = bias_variance_proba(mk, sample_train, X_test, p_test, n_rounds=20)
    bv["test AUC (one fit)"] = roc_auc_score(test["default"], mk().fit(*sample_train(0)).predict_proba(X_test)[:, 1])
    rows[name] = bv
bv_tab = pd.DataFrame(rows).T
print(bv_tab.to_string(float_format=lambda v: f"{v:.4f}"))
show("Noise floor (the best any model can do)", f"{bv_tab['noise'].iloc[0]:.4f}")
d, b = bv_tab.loc["deep tree (leaf >= 1)"], bv_tab.loc["BAGGING 100 deep trees"]
s, g = bv_tab.loc["stump (depth 1)"], bv_tab.loc["BOOSTING 200 stumps"]
note(f"""Each model was trained on 20 independent samples of {N_TRAIN:,} applicants and scored on
the same 3,000 test applicants whose TRUE default probability we know.
- Deep tree -> bagging: variance {d['variance']:.4f} -> {b['variance']:.4f} ({1 - b['variance'] / d['variance']:.0%} less), bias^2 barely moves.
- Stump -> boosting: bias^2 {s['bias^2']:.4f} -> {g['bias^2']:.4f} ({1 - g['bias^2'] / s['bias^2']:.0%} less), variance grows a little.
That is the whole theory of ensembles in two lines: bagging averages away
VARIANCE, boosting adds away BIAS. The best models balance both terms.""", keep_lines=True)

# %% =========================================================================
# LEVEL 3 (HARD): When each works, and why random forests decorrelate
# ============================================================================
banner("Unstable learners, correlated trees, and more trees", "HARD")

step("3.1  Bagging a STABLE learner does nothing")
rows = {}
for name, mk in {"logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                 "bagged logistic regression (100)": lambda: BaggingClassifier(
                     make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), n_estimators=100, n_jobs=-1, random_state=0),
                 "stump": lambda: DecisionTreeClassifier(max_depth=1),
                 "bagged stumps (100)": lambda: BaggingClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100,
                                                                  n_jobs=-1, random_state=0)}.items():
    rows[name] = bias_variance_proba(mk, sample_train, X_test, p_test, n_rounds=20)
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))
note("""Logistic regression and a stump barely change from one bootstrap sample to the
next, so averaging 100 near-identical models gives the same model: no variance
to remove, and bagging can't touch the bias. Bagging needs learners that are
UNSTABLE (deep trees), and boosting needs learners that are WEAK (shallow
trees). Choosing the wrong pairing wastes compute.""")

step("3.2  Why random forests pick random features: decorrelating the trees")
Xtr, ytr = sample_train(0)
rows = {}
for name, mf in {"bagging (all 11 features per split)": None, "max_features = 6": 6, "max_features = 3 ('sqrt')": 3,
                 "max_features = 1": 1}.items():
    rf = RandomForestClassifier(100, max_features=mf, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    per_tree = np.array([t.predict_proba(X_test.to_numpy())[:, 1] for t in rf.estimators_])
    sigma2 = per_tree.var(axis=0).mean()                                   # variance of ONE tree (across trees)
    corr = np.corrcoef(per_tree[:, :1000])[np.triu_indices(100, 1)].mean()  # average pairwise tree correlation
    rows[name] = {"avg correlation between trees": corr, "one tree's variance": sigma2,
                  "formula: rho s2 + (1-rho) s2 / B": corr * sigma2 + (1 - corr) * sigma2 / 100,
                  "test AUC": roc_auc_score(test["default"], rf.predict_proba(X_test)[:, 1])}
corr_tab = pd.DataFrame(rows).T
print(corr_tab.to_string(float_format=lambda v: f"{v:.4f}"))
note("""Averaging B trees whose predictions have variance s2 and pairwise correlation
rho leaves a variance of  rho s2 + (1 - rho) s2 / B. With many trees the second
term vanishes, so the correlation rho is the floor. Bagged trees all grab the
same strong features and are the MOST correlated; restricting each split to a
random subset of features DECORRELATES them and lowers the floor, at the price
of slightly weaker individual trees. Too few features (1) and the trees get
too weak; the default ('sqrt') is a good compromise.""")

step("3.3  Adding trees: bagging plateaus, boosting eventually over-fits")
rows = {}
for n in [1, 10, 50, 300]:
    bag = bias_variance_proba(lambda: RandomForestClassifier(n, n_jobs=-1, random_state=0), sample_train, X_test, p_test, n_rounds=8)
    bst = bias_variance_proba(lambda: GradientBoostingClassifier(max_depth=3, n_estimators=n, learning_rate=0.3, random_state=0),
                              sample_train, X_test, p_test, n_rounds=8)
    rows[f"{n} trees"] = {"forest bias^2": bag["bias^2"], "forest variance": bag["variance"],
                          "forest expected Brier": bag["expected Brier"], "boosting bias^2": bst["bias^2"],
                          "boosting variance": bst["variance"], "boosting expected Brier": bst["expected Brier"]}
grow = pd.DataFrame(rows).T
print(grow.to_string(float_format=lambda v: f"{v:.4f}"))
note(f"""A forest only ever adds more independent votes: its variance falls and then
plateaus, and its Brier never gets worse. Boosting (here with a fast learning
rate of 0.3) first removes bias, but every extra round also fits more noise:
past the sweet spot its variance keeps climbing and the expected Brier rises
again ({grow['boosting expected Brier'].min():.4f} at best, {grow['boosting expected Brier'].iloc[-1]:.4f} with 300 trees). That is why boosting
needs early stopping (Lab 05) and forests don't.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].scatter(x, y1, s=8, color="grey", alpha=0.6, label="noisy data")
    axes[0].plot(grid, f(grid[:, 0]), "k--", label="truth")
    for name, c in [("one deep tree", "#e45756"), ("bagging: 200 deep trees, averaged", "#4c78a8")]:
        axes[0].plot(grid, fits[name].predict(grid), color=c, label=name, lw=1.2)
    axes[0].set(title="Bagging removes variance")
    axes[0].legend(fontsize=7)
    axes[1].scatter(x, y1, s=8, color="grey", alpha=0.6)
    axes[1].plot(grid, f(grid[:, 0]), "k--")
    for name, c in [("one stump (1 question)", "#e45756"), ("boosting: 200 stumps, added", "#54a24b")]:
        axes[1].plot(grid, fits[name].predict(grid), color=c, label=name, lw=1.5)
    axes[1].set(title="Boosting removes bias")
    axes[1].legend(fontsize=7)
    names = list(bv_tab.index)
    axes[2].barh(names, bv_tab["bias^2"], color="#f58518", label="bias^2")
    axes[2].barh(names, bv_tab["variance"], left=bv_tab["bias^2"], color="#4c78a8", label="variance")
    axes[2].set(xlabel="error above the noise floor (Brier units)", title="Measured bias^2 + variance")
    axes[2].legend()
    axes[2].invert_yaxis()
    axes[2].tick_params(axis="y", labelsize=8)
    save_fig(fig, "03_bagging_vs_boosting")

banner("KEY TAKEAWAYS")
print("""  1. Bagging = deep trees on bootstrap samples, averaged: it removes variance.
  2. Boosting = small trees added sequentially, each fixing the remaining error: it removes bias.
  3. Bagging stable learners (logistic regression, stumps) changes nothing.
  4. Random forests decorrelate trees with random feature subsets: variance floor = rho x s2.
  5. More trees never hurt a forest; boosting over-fits after its best round.""")
