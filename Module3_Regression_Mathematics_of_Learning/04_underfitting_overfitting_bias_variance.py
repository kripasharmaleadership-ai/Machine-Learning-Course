"""
LAB 04 - Underfitting, Overfitting and the Bias-Variance Trade-off
==================================================================

THE SCENARIO
    A small utility predicts tomorrow's electricity demand from the temperature forecast.
    Demand is U-shaped (heating when cold, air-conditioning when hot). We KNOW the true
    curve (ml_lab_utils.energy_true_curve) and the noise level (sd = 12 MWh), so we can
    draw as many independent "histories" as we like and measure what no real project can.

THE THREE SOURCES OF ERROR (for squared error, exactly):
    expected test MSE = bias^2 + variance + irreducible noise

    bias^2   : how far the AVERAGE model (over many training sets) is from the truth.
               Too simple a model -> high bias -> UNDERFITTING.
    variance : how much the model changes when the training set changes.
               Too flexible a model -> high variance -> OVERFITTING.
    noise    : randomness no model can predict (here 12^2 = 144). The floor.

LEVELS
    EASY   : Fit polynomials of degree 1, 2, 4 and 15. Train vs test error.
    MEDIUM : Validation curves (choose the complexity) and learning curves (diagnose
             bias vs variance: does more data help?).
    HARD   : Measure bias^2 and variance directly from 300 training sets, for
             polynomial degree, tree depth and ridge penalty, and find that the best
             complexity depends on how much data you have.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml_lab_utils import (ENERGY_NOISE_SD, HAS_MPL, RANDOM_STATE, banner, energy_true_curve, make_energy_demand,
                          note, plt, save_fig, show, step)
from regression_toolkit import bias_variance_decomposition

NOISE_VAR = ENERGY_NOISE_SD ** 2
center = FunctionTransformer(lambda X: (np.asarray(X) - 14.0) / 22.0)      # temperature -> about [-1, 1]


def poly_model(degree, alpha=None):
    reg = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    return make_pipeline(center, PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


train = make_energy_demand(n=30, seed=RANDOM_STATE)
test = make_energy_demand(n=5000, seed=999)                    # a huge test set = the "true" test error
Xtr, ytr, Xte, yte = train[["temp_c"]], train.demand_mwh, test[["temp_c"]], test.demand_mwh

# %% =========================================================================
# LEVEL 1 (EASY): Too simple, just right, too flexible
# ============================================================================
banner("Underfitting vs overfitting: 30 days of data", "EASY")
rows, fitted = {}, {}
for d in [1, 2, 4, 15]:
    m = poly_model(d).fit(Xtr, ytr)
    fitted[d] = m
    rows[f"degree {d}"] = {"train RMSE": rmse(ytr, m.predict(Xtr)), "test RMSE": rmse(yte, m.predict(Xte))}
res = pd.DataFrame(rows).T
res["diagnosis"] = np.select(
    [res["train RMSE"] > 2 * ENERGY_NOISE_SD, res["test RMSE"] > 2 * res["train RMSE"]],
    ["UNDERFIT (high bias)", "OVERFIT (high variance)"], "good fit")
print(res.round(1).to_string())
show("Noise level = best possible RMSE", f"{ENERGY_NOISE_SD:.1f} MWh")
note("""Degree 1 is a straight line through a U: bad on training AND test data
(underfitting). Degree 15 has the LOWEST training error, below the noise level,
which is impossible without memorising noise. Its test error is the worst.
Rule of thumb: train error much lower than test error = overfitting; both high =
underfitting; both close to the noise level = a good fit.""")

if HAS_MPL:
    grid = np.linspace(-8, 36, 400)[:, None]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, (d, m) in zip(axes, fitted.items()):
        ax.scatter(Xtr, ytr, s=14, color="k", zorder=3, label="30 training days")
        ax.plot(grid, energy_true_curve(grid.ravel()), "g--", lw=1.5, label="true curve")
        ax.plot(grid, np.clip(m.predict(pd.DataFrame(grid, columns=["temp_c"])), 150, 700), color="#e45756",
                lw=2, label="model")
        ax.set(title=f"degree {d}: test RMSE {rows[f'degree {d}']['test RMSE']:.0f}", xlabel="temperature (C)",
               ylim=(180, 650))
    axes[0].set_ylabel("demand (MWh)")
    axes[0].legend(fontsize=7)
    save_fig(fig, "04_easy_under_over_fit")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Choosing complexity, diagnosing with learning curves
# ============================================================================
banner("Validation curves and learning curves", "MEDIUM")

step("2.1  Validation curve: 5-fold CV error for each degree (same 30 days)")
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
vc = {}
for d in range(1, 13):
    cv_rmse = -cross_val_score(poly_model(d), Xtr, ytr, cv=cv, scoring="neg_root_mean_squared_error")
    m = poly_model(d).fit(Xtr, ytr)
    vc[d] = {"train RMSE": rmse(ytr, m.predict(Xtr)), "CV RMSE": cv_rmse.mean(), "true test RMSE": rmse(yte, m.predict(Xte))}
vc = pd.DataFrame(vc).T
vc.index.name = "degree"
print(vc.round(1).to_string())
best_d = int(vc["CV RMSE"].idxmin())
show("Degree chosen by cross-validation", best_d)
show("Degree with the best true test RMSE", int(vc["true test RMSE"].idxmin()))
note("""Training error only goes down as the model gets more flexible, so it can never
choose the complexity. Cross-validation estimates the TEST error using only the
training data. With 30 rows it is noisy, and so is the true test column: one
sample of 30 days happens to suit degree 10, yet degree 12 on the same days
fails badly. CV picks a degree whose true error is within ~1 MWh of the luckiest
choice, and it flags the high-degree disasters (CV RMSE in the hundreds) that
the training error hides. A safe choice beats a lucky one (see 3.3).""")

step("2.2  Learning curves: does more data help?")
lc = {}
for n in [15, 30, 60, 120, 250, 500, 1000]:
    d_tr = make_energy_demand(n=n, seed=n)
    for name, d in [("degree 1 (simple)", 1), ("degree 12 (flexible)", 12)]:
        m = poly_model(d).fit(d_tr[["temp_c"]], d_tr.demand_mwh)
        lc.setdefault(name, {})[n] = (rmse(d_tr.demand_mwh, m.predict(d_tr[["temp_c"]])), rmse(yte, m.predict(Xte)))
lc_tab = pd.DataFrame({f"{k} {t}": {n: v[i] for n, v in d.items()} for k, d in lc.items()
                       for i, t in enumerate(["train", "test"])})
lc_tab.index.name = "training days"
print(lc_tab.round(1).to_string())
note("""HIGH BIAS (degree 1): train and test errors meet quickly, at a level far above
the noise (12). More data does NOT help: the model simply cannot bend. Fix: a
more flexible model or better features. HIGH VARIANCE (degree 12): a large gap
between train and test that SHRINKS as data grows. Fix: more data,
regularisation (Lab 05), or a simpler model.""")
print("""
  Diagnosis cheat-sheet
  ---------------------------------------------------------------------------
  train error high, test error high, small gap  -> underfitting  -> more capacity / features
  train error low,  test error high, big gap    -> overfitting   -> data, regularise, simplify
  both close to the noise level                 -> done; further gains need new INFORMATION""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(vc.index, vc["train RMSE"], "o-", label="train")
    axes[0].plot(vc.index, vc["CV RMSE"], "s-", label="5-fold CV")
    axes[0].plot(vc.index, vc["true test RMSE"], "^-", label="true test")
    axes[0].axhline(ENERGY_NOISE_SD, color="k", ls=":", label="noise floor")
    axes[0].set(yscale="log", xlabel="polynomial degree", ylabel="RMSE (MWh)", title="Validation curve (30 days)")
    axes[0].legend(fontsize=8)
    for k, d in lc.items():
        ns = list(d)
        axes[1].plot(ns, [v[1] for v in d.values()], "o-", label=f"{k} test")
        axes[1].plot(ns, [v[0] for v in d.values()], "o--", label=f"{k} train", alpha=0.6)
    axes[1].axhline(ENERGY_NOISE_SD, color="k", ls=":")
    axes[1].set(xscale="log", yscale="log", xlabel="training days", ylabel="RMSE", title="Learning curves")
    axes[1].legend(fontsize=7)
    save_fig(fig, "04_medium_validation_learning_curves")

# %% =========================================================================
# LEVEL 3 (HARD): Measure bias and variance directly
# ============================================================================
banner("Bias^2 + variance + noise, measured from 300 training sets", "HARD")
x_eval = np.linspace(-6, 34, 200)[:, None]                 # stay inside the training range
f_eval = energy_true_curve(x_eval.ravel())
X_eval = pd.DataFrame(x_eval, columns=["temp_c"])


def sampler(n):
    def sample(seed):
        d = make_energy_demand(n=n, seed=10_000 + seed)
        return d[["temp_c"]], d.demand_mwh
    return sample


def decompose(make_model, n=30, rounds=300):
    out = bias_variance_decomposition(make_model, sampler(n), X_eval, f_eval, NOISE_VAR, n_rounds=rounds)
    out.pop("_preds")
    return out


step("3.1  Polynomial degree (30 training days)")
bv_deg = pd.DataFrame({d: decompose(lambda d=d: poly_model(d)) for d in [1, 2, 3, 4, 6, 8, 10]}).T
bv_deg.index.name = "degree"
print(bv_deg.round(1).to_string())
note("""Read across a row: bias^2 + variance + noise = expected test MSE, an exact
identity, and we measured each piece. Moving down the table, bias^2 falls (the
curve can bend into the U) and variance rises (the curve starts chasing each
training set's noise). The best degree is where the SUM is smallest, not where
either piece is zero.""")

step("3.2  The same trade-off in two other knobs")
bv_tree = pd.DataFrame({f"max_depth={dpt}": decompose(lambda dpt=dpt: DecisionTreeRegressor(max_depth=dpt,
                                                                                              random_state=0))
                        for dpt in [1, 2, 3, 5, None]}).T
bv_ridge = pd.DataFrame({f"alpha={a:g}": decompose(lambda a=a: poly_model(10, alpha=a))
                         for a in [1e-6, 1e-3, 0.1, 1, 10, 100]}).T
print(bv_tree.round(1).to_string())
print()
print(bv_ridge.round(1).to_string())
note("""Tree depth: a stump (depth 1) predicts two values: mostly bias. With only 30
points, depth 5 already isolates almost every point, and an unlimited tree
memorises all of them: almost pure variance. Ridge on a degree-10 polynomial: the
penalty alpha is a CONTINUOUS complexity dial: tiny alpha = the wild degree-10
fit, huge alpha = almost flat. A moderate alpha keeps the flexible features but
stops them from chasing noise. That is regularisation (Lab 05).""")

step("3.3  Plot twist: the best model depends on how much data you have")
best = {}
for n in [12, 30, 100, 1000]:
    tab = {d: decompose(lambda d=d: poly_model(d), n=n, rounds=150)["expected MSE"] for d in [1, 2, 3, 4, 6, 8, 10]}
    s = pd.Series(tab)
    best[n] = {"best degree": int(s.idxmin()), "its RMSE": np.sqrt(s.min()), "degree 10 RMSE": np.sqrt(s[10]),
               "degree 2 RMSE": np.sqrt(s[2])}
best = pd.DataFrame(best).T
best.index.name = "training days"
print(best.round(1).to_string())
note("""With 12 days, a crude quadratic beats a degree-10 curve that could represent the
truth far better: there is not enough data to pin its 10 coefficients down. With
1,000 days, the flexible model wins and gets close to the noise floor (12). There
is no 'best model' in the abstract, only the best model FOR THIS AMOUNT OF DATA.
This is why simple models often win in small-data business problems, and why
huge neural nets need huge datasets.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(bv_deg.index, bv_deg["bias^2"], "o-", label="bias^2")
    axes[0].plot(bv_deg.index, bv_deg["variance"], "s-", label="variance")
    axes[0].plot(bv_deg.index, bv_deg["expected MSE"], "k^-", label="expected test MSE")
    axes[0].axhline(NOISE_VAR, color="grey", ls=":", label="noise")
    axes[0].set(yscale="log", xlabel="polynomial degree", ylabel="MSE", title="Bias-variance trade-off (30 days)")
    axes[0].legend(fontsize=8)
    preds = bias_variance_decomposition(lambda: poly_model(8), sampler(30), X_eval, f_eval, NOISE_VAR, 25)["_preds"]
    for p in preds:
        axes[1].plot(x_eval, p, color="#e45756", alpha=0.25, lw=1)
    axes[1].plot(x_eval, f_eval, "g", lw=2.5, label="true curve")
    axes[1].set(ylim=(180, 650), xlabel="temperature (C)", title="Degree 8 trained on 25 different 30-day samples")
    axes[1].legend()
    save_fig(fig, "04_hard_bias_variance")

banner("KEY TAKEAWAYS")
print("""  1. Underfitting = high bias (too simple); overfitting = high variance (too flexible).
  2. Training error always falls with complexity: choose complexity with validation / CV.
  3. Learning curves tell you whether more data will help (variance) or not (bias).
  4. Expected test MSE = bias^2 + variance + noise. You can only trade the first two.
  5. The right complexity depends on the amount of data: small data favours simple models.""")
