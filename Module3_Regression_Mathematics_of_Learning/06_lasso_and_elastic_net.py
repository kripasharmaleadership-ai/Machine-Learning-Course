"""
LAB 06 - Lasso and Elastic Net
==============================

THE PENALTIES
    Ridge (L2)  : MSE + alpha * sum(w_j^2)                         shrinks, never exactly 0
    Lasso (L1)  : MSE + alpha * sum(|w_j|)                         shrinks AND sets some w_j = 0
    Elastic Net : MSE + alpha * (l1_ratio * sum|w_j| + (1 - l1_ratio)/2 * sum w_j^2)

WHY L1 GIVES EXACT ZEROS
    The gradient of w^2 vanishes near 0 (the push towards zero gets weaker and weaker).
    The "gradient" of |w| is a constant +-1: a feature must earn at least alpha of
    error reduction to keep a non-zero weight. Weak features are switched OFF.
    -> Lasso does feature SELECTION and estimation in one step.

WHY ELASTIC NET
    With correlated features, Lasso picks ONE of the group more or less at random
    and zeros the rest. Elastic Net's L2 part makes correlated features share the
    weight (like Ridge), while the L1 part still removes the irrelevant ones.

LEVELS
    EASY   : Lasso on house prices + 20 junk columns: the junk disappears.
    MEDIUM : Coefficient paths (Ridge vs Lasso vs Elastic Net), soft-thresholding,
             LassoCV and the one-standard-error rule.
    HARD   : 500 features, 100 rows (more features than data): which method recovers the
             truth, and why "Lasso dropped it" does not mean "it doesn't matter".
"""
# %%
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, HOUSE_FEATURES, RANDOM_STATE, banner, make_house_sales, note, plt, save_fig,
                          show, step)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% =========================================================================
# LEVEL 1 (EASY): Automatic feature selection
# ============================================================================
banner("Lasso switches junk features off", "EASY")
houses = make_house_sales(n=3000, seed=12)
rng = np.random.default_rng(0)
junk = pd.DataFrame(rng.normal(0, 1, (len(houses), 20)), columns=[f"junk_{i:02d}" for i in range(20)])
junk["listing_photos"] = rng.integers(5, 40, len(houses))           # plausible-sounding, but irrelevant
data = pd.concat([houses, junk], axis=1)
FEATS = HOUSE_FEATURES + list(junk.columns)
tr, te = train_test_split(data, test_size=0.3, random_state=RANDOM_STATE)

rows = {}
for name, m in {"OLS": LinearRegression(), "Ridge (alpha=10)": Ridge(alpha=10),
                "Lasso (alpha=1,500)": Lasso(alpha=1500, max_iter=50000)}.items():
    p = make_pipeline(StandardScaler(), m).fit(tr[FEATS], tr.price)
    coef = pd.Series(p[-1].coef_, index=FEATS)
    rows[name] = {"non-zero coefficients": int((coef.abs() > 1e-6).sum()),
                  "non-zero JUNK coefficients": int((coef[junk.columns].abs() > 1e-6).sum()),
                  "largest |junk coef| ($)": coef[junk.columns].abs().max(),
                  "test RMSE ($)": np.sqrt(np.mean((te.price - p.predict(te[FEATS])) ** 2))}
print(pd.DataFrame(rows).T.round(0).to_string())
lasso = make_pipeline(StandardScaler(), Lasso(alpha=1500, max_iter=50000)).fit(tr[FEATS], tr.price)
print("\n  Lasso keeps:", [f for f, c in zip(FEATS, lasso[-1].coef_) if abs(c) > 1e-6])
note("""OLS and Ridge give every junk column a small non-zero weight: each one is noise
the model has learned. Lasso sets them to EXACTLY zero and keeps the 6 real
drivers. A smaller model is cheaper to serve, needs fewer inputs from the
upstream systems, and is easier to explain. Here the accuracy gain is small
because 2,100 rows are enough to average 21 junk columns away. The HARD level
shows what happens when they are not.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Paths, the mechanism, and choosing alpha
# ============================================================================
banner("How L1 differs from L2", "MEDIUM")

step("2.1  Coefficient paths for the 6 real features + 3 junk")
show_feats = HOUSE_FEATURES + ["junk_00", "junk_01", "listing_photos"]
sc = StandardScaler().fit(tr[show_feats])
Xs, ys = sc.transform(tr[show_feats]), tr.price.to_numpy()
alphas_l = np.logspace(1, 5, 40)
paths = {"Ridge": [Ridge(alpha=a * 2 / 30).fit(Xs, ys).coef_ for a in alphas_l],
         "Lasso": [Lasso(alpha=a, max_iter=50000).fit(Xs, ys).coef_ for a in alphas_l],
         "Elastic Net (l1_ratio=0.5)": [ElasticNet(alpha=a, l1_ratio=0.5, max_iter=50000).fit(Xs, ys).coef_
                                        for a in alphas_l]}
nz = pd.DataFrame({k: [(np.abs(c) > 1e-6).sum() for c in v] for k, v in paths.items()}, index=alphas_l.round(0))
nz.index.name = "alpha"
print(nz.iloc[::6].to_string())
drop_order = pd.Series({f: alphas_l[np.argmax(np.abs(np.array(paths["Lasso"])[:, i]) < 1e-6)]
                        if (np.abs(np.array(paths["Lasso"])[:, i]) < 1e-6).any() else np.inf
                        for i, f in enumerate(show_feats)}).sort_values()
print("\n  Alpha at which Lasso drops each feature (junk first, strongest last):")
print(drop_order.round(0).to_string())
note("""Lasso removes features one by one as alpha grows, irrelevant ones first, and
the ORDER is a ranking of how much each feature is worth. Ridge (same plot,
smooth curves) never drops anything. Elastic Net sits in between.""")

step("2.2  The mechanism: soft-thresholding")
w_ols = np.array([-3.0, -1.2, -0.4, 0.0, 0.3, 0.9, 2.5])
lam = 1.0
print(pd.DataFrame({"OLS coefficient": w_ols,
                    "Ridge: w / (1 + lam)": w_ols / (1 + lam),
                    "Lasso: sign(w) * max(|w| - lam, 0)": np.sign(w_ols) * np.maximum(np.abs(w_ols) - lam, 0)}
                   ).to_string(index=False))
note("""For uncorrelated (orthogonal) features both penalties have a closed form. Ridge
divides every coefficient by the same factor: big ones stay big, small ones get
small but not zero. Lasso SUBTRACTS a fixed amount and clips at zero: anything
smaller than the threshold is gone. That is the whole difference.""")

step("2.3  Choosing alpha: LassoCV and the one-standard-error rule")
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
lcv = make_pipeline(StandardScaler(), LassoCV(cv=cv, alphas=60, max_iter=50000)).fit(tr[FEATS], tr.price)
L = lcv[-1]
mse_mean = L.mse_path_.mean(axis=1)
mse_se = L.mse_path_.std(axis=1) / np.sqrt(L.mse_path_.shape[1])
i_min = int(np.argmin(mse_mean))
i_1se = int(np.where(mse_mean <= mse_mean[i_min] + mse_se[i_min])[0].min())   # largest alpha within 1 SE
rows = {}
for label, a in [("alpha_min (lowest CV error)", L.alphas_[i_min]), ("alpha_1se (simplest within 1 SE)", L.alphas_[i_1se])]:
    m = make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=50000)).fit(tr[FEATS], tr.price)
    rows[label] = {"alpha": a, "non-zero": int((np.abs(m[-1].coef_) > 1e-6).sum()),
                   "CV RMSE": np.sqrt(mse_mean[L.alphas_ == a][0]),
                   "test RMSE": np.sqrt(np.mean((te.price - m.predict(te[FEATS])) ** 2))}
print(pd.DataFrame(rows).T.round(1).to_string())
note("""The alpha with the lowest CV error is permissive: it keeps most of the junk,
because each junk column buys a tiny, noisy CV improvement. The one-standard-
error rule picks the SIMPLEST model whose CV error is within one standard error
of the best. Here it keeps 5 features and is ~1.5% worse on test: honest
trade-off, it also dropped 'bedrooms', a REAL but weak driver (-$9k per bedroom
holding size fixed). Use the 1-SE rule when a smaller model has business value
(fewer inputs to collect, easier to explain), and check what it removed.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, (name, P) in zip(axes, paths.items()):
        P = np.array(P)
        for i, f in enumerate(show_feats):
            ax.plot(alphas_l, P[:, i], ls="--" if f.startswith(("junk", "listing")) else "-", label=f)
        ax.set(xscale="log", title=name, xlabel="alpha (penalty strength)")
    axes[0].set_ylabel("coefficient ($ per sd)")
    axes[0].legend(fontsize=6)
    save_fig(fig, "06_medium_paths_l1_l2")

# %% =========================================================================
# LEVEL 3 (HARD): More features than rows
# ============================================================================
banner("p = 500 features, n = 100 rows", "HARD")
# A biotech start-up predicts a patient's drug response from 500 gene-expression levels.
# Truth: 10 genes matter, in 5 CORRELATED PAIRS (co-regulated genes). 490 are noise.
rng = np.random.default_rng(7)
n_tr, n_te, p = 100, 3000, 500
true_idx = np.arange(10)
beta = np.zeros(p)
beta[true_idx] = [3, 3, -2.5, -2.5, 2, 2, 1.5, 1.5, -1.5, -1.5]


def genes(n):
    X = rng.normal(0, 1, (n, p))
    for k in range(0, 10, 2):                                      # partner gene = 0.9 correlated
        X[:, k + 1] = 0.9 * X[:, k] + np.sqrt(1 - 0.9 ** 2) * rng.normal(0, 1, n)
    return X, X @ beta + rng.normal(0, 2.0, n)


G_tr, r_tr = genes(n_tr)
G_te, r_te = genes(n_te)
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
models = {
    "OLS (min-norm solution)": LinearRegression(),
    "RidgeCV (5-fold)": RidgeCV(alphas=np.logspace(-2, 5, 40), cv=cv),
    "LassoCV": LassoCV(cv=cv, max_iter=50000, random_state=0),
    "ElasticNetCV": ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=cv, max_iter=50000, random_state=0),
}
rows, fitted = {}, {}
for name, m in models.items():
    m.fit(G_tr, r_tr)
    fitted[name] = m
    sel = np.abs(m.coef_) > 1e-6
    tp = sel[true_idx].sum()
    rows[name] = {"train R^2": m.score(G_tr, r_tr), "test R^2": m.score(G_te, r_te),
                  "features used": int(sel.sum()), "true genes found (of 10)": int(tp),
                  "precision of selection": tp / max(sel.sum(), 1)}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""With 500 unknowns and 100 equations, OLS has infinitely many perfect solutions
(train R^2 = 1.0); sklearn returns the smallest one, which explains less than
half of the test variance. Honest result: Ridge cannot improve on it. CV picks
alpha ~ 0, and every larger alpha is WORSE, because spreading weight evenly over
500 genes is the wrong assumption when only 10 matter. Lasso and Elastic Net bet
on SPARSITY, which is true here: test R^2 > 0.9, and a short list of genes for
the lab to validate. Sparse truth -> L1 wins; dense truth (Lab 05, 3.4) -> L2
wins. Elastic Net is the safe middle.""")

step("3.1  Correlated pairs: Lasso may pick one, Elastic Net shares")
pair_rows = {}
for name in ["LassoCV", "ElasticNetCV"]:
    c = fitted[name].coef_
    pair_rows[name] = {f"gene {k} & {k + 1}": f"{c[k]:+.2f} / {c[k + 1]:+.2f}" for k in range(0, 10, 2)}
pair_rows["TRUE"] = {f"gene {k} & {k + 1}": f"{beta[k]:+.2f} / {beta[k + 1]:+.2f}" for k in range(0, 10, 2)}
print(pd.DataFrame(pair_rows).to_string())

step("3.2  Selection stability over 100 bootstrap samples")
counts = {"Lasso": np.zeros(p), "Elastic Net": np.zeros(p)}
a_l, a_e, r_e = fitted["LassoCV"].alpha_, fitted["ElasticNetCV"].alpha_, fitted["ElasticNetCV"].l1_ratio_
for b in range(100):
    i = rng.integers(0, n_tr, n_tr)
    counts["Lasso"] += np.abs(Lasso(alpha=a_l, max_iter=20000).fit(G_tr[i], r_tr[i]).coef_) > 1e-6
    counts["Elastic Net"] += np.abs(ElasticNet(alpha=a_e, l1_ratio=r_e, max_iter=20000).fit(G_tr[i], r_tr[i]).coef_) > 1e-6
stab = pd.DataFrame({k: {**{f"gene {g}": v[g] for g in true_idx},
                         "noise genes selected >50% of the time": int((v[10:] > 50).sum())} for k, v in counts.items()})
print(stab.T.astype(int).to_string())
weak = int(np.argmin(counts["Lasso"][true_idx]))
partner = weak + 1 if weak % 2 == 0 else weak - 1
note(f"""Plot twist: gene {weak} truly matters (coefficient {beta[weak]:+.1f}), but Lasso keeps it in
only {counts['Lasso'][weak]:.0f}% of the bootstrap samples: its 0.9-correlated partner gene {partner}
absorbs its effect. A scientist reading the Lasso output would conclude 'gene
{weak} has no effect' and drop it from the follow-up study. Elastic Net keeps it in
{counts['Elastic Net'][weak]:.0f}% of samples, at the price of letting a few more noise genes in.
Rule: a zero Lasso coefficient means 'not needed GIVEN the other features',
never 'no effect'. Check selection stability before acting on it.""")

banner("KEY TAKEAWAYS")
print("""  1. Lasso (L1) shrinks AND selects: weak features get exactly zero weight.
  2. The order in which Lasso drops features is a usable importance ranking.
  3. Use the one-standard-error rule when a smaller model has value.
  4. Sparse truth (few drivers) -> Lasso / Elastic Net; dense truth (many small effects) -> Ridge.
  5. Correlated features: Lasso picks one arbitrarily; Elastic Net keeps the group. Zero != no effect.""")
