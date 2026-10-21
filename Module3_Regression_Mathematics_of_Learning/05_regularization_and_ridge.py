"""
LAB 05 - Regularization and Ridge Regression
============================================

THE IDEA
    Overfitting models have WILD coefficients: huge positive and negative weights that
    cancel out on the training data and explode everywhere else. Regularisation adds
    a price for large coefficients to the cost function:

        Ridge (L2):  J(w) = MSE(w) + alpha * sum(w_j^2)

    alpha = 0       -> ordinary least squares
    alpha -> inf    -> all coefficients -> 0 (predict the mean)
    in between      -> a little bias, much less variance (Lab 04)

    Closed form:  w = (X^T X + alpha * I)^-1 X^T y      (always invertible for alpha > 0)
    The intercept is NOT penalised, and features MUST be standardised (the penalty
    treats all coefficients alike, so their units must be comparable).

LEVELS
    EASY   : Tame a degree-12 polynomial; ridge by hand vs sklearn.
    MEDIUM : Coefficient paths, choosing alpha by cross-validation, collinearity, and
             why scaling is mandatory.
    HARD   : Ridge = least squares on "fake data" = weight decay; effective degrees of
             freedom; and the wide-data rescue: 60 correlated features, 80 patients.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

from ml_lab_utils import (HAS_MPL, HOUSE_FEATURES, RANDOM_STATE, banner, make_energy_demand, make_house_sales,
                          note, plt, save_fig, show, step)
from regression_toolkit import LinearRegressionGD

ALPHAS = np.logspace(-4, 4, 41)

# %% =========================================================================
# LEVEL 1 (EASY): Shrinking wild coefficients
# ============================================================================
banner("Ridge tames an overfitting polynomial", "EASY")
train = make_energy_demand(n=20, seed=3)
test = make_energy_demand(n=5000, seed=999)
to_poly = make_pipeline(FunctionTransformer(lambda X: (np.asarray(X) - 14) / 22),
                        PolynomialFeatures(12, include_bias=False), StandardScaler())
P_tr = to_poly.fit_transform(train[["temp_c"]])
P_te = to_poly.transform(test[["temp_c"]])
ytr, yte = train.demand_mwh.to_numpy(), test.demand_mwh.to_numpy()

rows = {}
for a in [0, 1e-4, 1e-2, 1, 100]:
    m = LinearRegression() if a == 0 else Ridge(alpha=a)
    m.fit(P_tr, ytr)
    rows[f"alpha={a:g}"] = {"largest |coef|": np.abs(m.coef_).max(), "sum coef^2": np.sum(m.coef_ ** 2),
                            "train RMSE": np.sqrt(np.mean((ytr - m.predict(P_tr)) ** 2)),
                            "test RMSE": np.sqrt(np.mean((yte - m.predict(P_te)) ** 2))}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:,.1f}"))
note("""Without a penalty the 12 coefficients reach ~100,000, fighting each other
to thread the curve through 20 noisy points. A small alpha shrinks them by orders
of magnitude: training error rises a little, test error falls dramatically. Too
much penalty flattens the curve again (underfitting). Same features, same data:
alpha alone moves the model along the bias-variance trade-off.""")

step("1.1  Ridge by hand: w = (X^T X + alpha I)^-1 X^T y on centred data")
alpha = 1.0
Xc, yc = P_tr - P_tr.mean(0), ytr - ytr.mean()                 # centring = not penalising the intercept
w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ yc)
b = ytr.mean() - P_tr.mean(0) @ w
sk = Ridge(alpha=alpha).fit(P_tr, ytr)
show("Max |difference| in coefficients vs sklearn", f"{np.abs(w - sk.coef_).max():.2e}")
show("Intercept: by hand vs sklearn", f"{b:.3f} vs {sk.intercept_:.3f}")
note("""Adding alpha to the diagonal of X^T X is the entire algorithm. It also makes the
matrix ALWAYS invertible: Ridge works even with more features than rows, or with
perfectly duplicated columns, where ordinary least squares has no unique answer.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Paths, choosing alpha, collinearity, scaling
# ============================================================================
banner("Using Ridge properly", "MEDIUM")
houses = make_house_sales(n=600, seed=8)
houses["sqm_appraiser"] = houses.sqft / 10.764 + np.random.default_rng(3).normal(0, 4, len(houses))
FEATS = HOUSE_FEATURES + ["sqm_appraiser"]
h_tr, h_te = train_test_split(houses, test_size=0.3, random_state=RANDOM_STATE)
sc = StandardScaler().fit(h_tr[FEATS])
H_tr, H_te = sc.transform(h_tr[FEATS]), sc.transform(h_te[FEATS])
yh_tr, yh_te = h_tr.price.to_numpy(), h_te.price.to_numpy()

step("2.1  Coefficient path: every coefficient as alpha grows")
path = pd.DataFrame([Ridge(alpha=a).fit(H_tr, yh_tr).coef_ for a in ALPHAS], index=ALPHAS, columns=FEATS)
print(path.iloc[::8].round(0).to_string())
note("""Read top to bottom: as alpha grows every coefficient shrinks smoothly towards 0,
but none becomes EXACTLY 0 (that is Lasso, Lab 06). The two size columns, which
are near-duplicates, quickly settle to SHARING the size effect.""")

step("2.2  Collinearity: Ridge stabilises the coefficients")
rng = np.random.default_rng(1)
boot = {"OLS": [], "Ridge alpha=10": []}
for _ in range(300):
    i = rng.integers(0, len(H_tr), len(H_tr))
    boot["OLS"].append(LinearRegression().fit(H_tr[i], yh_tr[i]).coef_[[0, -1]])
    boot["Ridge alpha=10"].append(Ridge(alpha=10).fit(H_tr[i], yh_tr[i]).coef_[[0, -1]])
stab = pd.DataFrame({k: {"sqft coef: mean": np.mean(v, 0)[0], "sqft coef: sd": np.std(v, 0)[0],
                         "sqm coef: mean": np.mean(v, 0)[1], "sqm coef: sd": np.std(v, 0)[1]}
                     for k, v in boot.items()}).T
print(stab.round(0).to_string())
for name, m in [("OLS", LinearRegression()), ("Ridge alpha=10", Ridge(alpha=10))]:
    show(f"Test RMSE {name}", f"${np.sqrt(np.mean((yh_te - m.fit(H_tr, yh_tr).predict(H_te)) ** 2)):,.0f}")
note("""The bootstrap standard deviation of each size coefficient drops by about 10x.
Ridge splits a shared effect EVENLY between correlated features instead of
letting the data flip a coin, at almost no cost in accuracy. When
features are correlated (they almost always are), Ridge coefficients are the
ones you can show to a stakeholder.""")

step("2.3  Choosing alpha with cross-validation")
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
cv_rmse = [-cross_val_score(make_pipeline(StandardScaler(), Ridge(alpha=a)), h_tr[FEATS], yh_tr, cv=cv,
                            scoring="neg_root_mean_squared_error").mean() for a in ALPHAS]
best_a = ALPHAS[int(np.argmin(cv_rmse))]
rcv = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(h_tr[FEATS], yh_tr)
show("alpha with lowest 5-fold CV RMSE", f"{best_a:.3g}")
show("RidgeCV (efficient leave-one-out) alpha", f"{rcv[-1].alpha_:.3g}")
show("CV RMSE at alpha = 1e-4 / best / 1e4", f"${cv_rmse[0]:,.0f} / ${min(cv_rmse):,.0f} / ${cv_rmse[-1]:,.0f}")
note("""alpha is a HYPERPARAMETER: it is chosen with validation data, never by
looking at the test set (Module 1, Lab 07). RidgeCV does this efficiently
(leave-one-out has a closed form for Ridge). Here the data are plentiful and the
features few, so the best alpha is small and the gain is modest. Ridge earns its
keep when data are scarce or features many (3.3).""")

step("2.4  Why scaling is MANDATORY for Ridge")
alt_tr = h_tr.assign(sqft=h_tr.sqft / 1000)                    # same information, sqft in thousands
alt_te = h_te.assign(sqft=h_te.sqft / 1000)
rows = {}
for name, (a_tr, a_te, unit) in {"sqft in sqft": (h_tr, h_te, 1),
                                 "sqft in thousands": (alt_tr, alt_te, 1000)}.items():
    raw = Ridge(alpha=10).fit(a_tr[HOUSE_FEATURES], yh_tr)
    scl = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(a_tr[HOUSE_FEATURES], yh_tr)
    rows[name] = {"no scaling: learned $ per sqft": raw.coef_[0] / unit,
                  "no scaling: test RMSE": np.sqrt(np.mean((yh_te - raw.predict(a_te[HOUSE_FEATURES])) ** 2)),
                  "StandardScaler: test RMSE": np.sqrt(np.mean((yh_te - scl.predict(a_te[HOUSE_FEATURES])) ** 2))}
print(pd.DataFrame(rows).T.round(1).to_string())
note("""Same houses, same alpha, the only change is that someone stored sqft in
thousands. Without scaling, the model now believes a square foot is worth ~25%
less: in thousands the coefficient must be 1000x larger, so the same penalty
shrinks it far more. With a StandardScaler inside the pipeline the model is
unit-invariant. Ridge, Lasso, Elastic Net, kNN, SVMs and neural nets
all need this.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for c in FEATS:
        axes[0].plot(ALPHAS, path[c], label=c)
    axes[0].set(xscale="log", xlabel="alpha", ylabel="coefficient ($ per sd)", title="Ridge coefficient paths")
    axes[0].legend(fontsize=7)
    axes[1].plot(ALPHAS, cv_rmse, "o-", ms=3)
    axes[1].axvline(best_a, color="r", ls="--", label=f"best alpha = {best_a:.3g}")
    axes[1].set(xscale="log", xlabel="alpha", ylabel="5-fold CV RMSE ($)", title="Choosing alpha")
    axes[1].legend()
    save_fig(fig, "05_medium_ridge_paths")

# %% =========================================================================
# LEVEL 3 (HARD): What Ridge really is, and where it shines
# ============================================================================
banner("Three views of Ridge, and the wide-data rescue", "HARD")

step("3.1  Ridge = ordinary least squares on FAKE data")
alpha = 5.0
Xc = H_tr - H_tr.mean(0)
yc = yh_tr - yh_tr.mean()
X_aug = np.vstack([Xc, np.sqrt(alpha) * np.eye(Xc.shape[1])])      # one fake row per feature...
y_aug = np.r_[yc, np.zeros(Xc.shape[1])]                            # ...whose target is 0
w_aug = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]
w_ridge = Ridge(alpha=alpha).fit(H_tr, yh_tr).coef_
show("Max |OLS on augmented data - Ridge|", f"{np.abs(w_aug - w_ridge).max():.2e}")
note("""Append p fake rows that say 'this feature alone predicts 0': least squares on the
augmented data IS Ridge. A penalty is a form of prior knowledge ('effects are
probably small'), worth as much as a few extra data points. That is why it
matters most when real data is scarce.""")

step("3.2  Ridge = weight decay in gradient descent")
gd = LinearRegressionGD(learning_rate=0.1, n_epochs=20000, alpha=alpha / len(H_tr), tol=1e-14,
                        patience=50).fit(H_tr, yh_tr / 1000)
show("Max |GD with weight decay - Ridge| ($k)", f"{np.abs(gd.coef_ - w_ridge / 1000).max():.2e}")
note("""Gradient descent on MSE + alpha * ||w||^2 has the update
w <- w - lr * (gradient + 2 alpha w) = (1 - 2 lr alpha) w - lr * gradient: every
step first DECAYS the weights a little. Deep-learning optimisers call this
'weight decay', and LLMs are trained with it (Lab 10). (sklearn's Ridge uses the
SUM of squared errors, hence alpha / n in our MSE-based class.)""")

step("3.3  Effective degrees of freedom: complexity as a continuous number")
d = np.linalg.svd(Xc, compute_uv=False)
for a in [0, 1, 10, 100, 1000, 10000]:
    show(f"alpha = {a:>6}: effective number of parameters", f"{np.sum(d ** 2 / (d ** 2 + a)):.2f} of {len(d)}")
note("""df(alpha) = sum d_j^2 / (d_j^2 + alpha), using the singular values d_j of X.
OLS spends all 7 parameters; Ridge 'uses' a fraction of each, starting with the
directions the data says least about. Model complexity is not just 'how many
features', it is how freely their coefficients can move.""")

step("3.4  Plot twist: 80 patients, 60 correlated lab measurements")
rng = np.random.default_rng(21)
n_tr, n_te, p, k = 80, 2000, 60, 4
load = rng.normal(0, 1, (k, p))                                  # 4 hidden health factors drive 60 lab tests


def patients(n):
    z = rng.normal(0, 1, (n, k))
    labs = z @ load + rng.normal(0, 0.8, (n, p))                  # each lab test = factors + noise
    cost = 9000 + z @ np.array([2500, 1800, -1200, 900]) + rng.normal(0, 1500, n)
    return labs, cost


L_tr, c_tr = patients(n_tr)
L_te, c_te = patients(n_te)
rows = {}
for name, m in {"OLS (60 coefficients, 80 rows)": make_pipeline(StandardScaler(), LinearRegression()),
                "RidgeCV": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
                "mean only (baseline)": None}.items():
    if m is None:
        pred_tr, pred_te = np.full(n_tr, c_tr.mean()), np.full(n_te, c_tr.mean())
    else:
        m.fit(L_tr, c_tr)
        pred_tr, pred_te = m.predict(L_tr), m.predict(L_te)
    rows[name] = {"train R^2": 1 - np.mean((c_tr - pred_tr) ** 2) / np.var(c_tr),
                  "test R^2": 1 - np.mean((c_te - pred_te) ** 2) / np.var(c_te),
                  "test RMSE $": np.sqrt(np.mean((c_te - pred_te) ** 2))}
print(pd.DataFrame(rows).T.round(3).to_string())
ridge_alpha = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(L_tr, c_tr)[-1].alpha_
show("alpha chosen by RidgeCV", f"{ridge_alpha:.3g}")
note("""With nearly as many features as patients, OLS fits the training data almost
perfectly and generalises poorly: each lab test gets a confident, wrong weight.
Ridge recognises that the 60 tests are noisy copies of a few underlying
factors and spreads the weight across each correlated group, which averages the
noise away. Same data, same features: one line of code lifts the test R^2
from ~0.54 to ~0.82 and cuts the error by a third. Wide, correlated data (sensors, lab panels, survey items, pixels,
text) is where Ridge is the default.""")

banner("KEY TAKEAWAYS")
print("""  1. Regularisation = MSE + a penalty on coefficient size; alpha dials complexity continuously.
  2. Ridge (L2) shrinks all coefficients smoothly, never to exactly zero.
  3. Choose alpha by cross-validation; ALWAYS standardise features first; don't penalise the intercept.
  4. Ridge stabilises correlated coefficients and works even when features >= rows.
  5. Ridge = least squares + fake data = weight decay. Neural nets use the same idea.""")
