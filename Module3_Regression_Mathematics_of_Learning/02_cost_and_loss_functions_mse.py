"""
LAB 02 - Cost and Loss Functions, Mean Squared Error
====================================================

VOCABULARY
    residual   e_i = y_i - y_hat_i                (one prediction's error)
    loss       L(y_i, y_hat_i), e.g. e_i^2        (the penalty for ONE example)
    cost       J(params) = average loss            (one number for the whole dataset)
    Training = searching for the parameters that make the COST as small as possible.

THE MOST COMMON COST: MEAN SQUARED ERROR
    MSE = (1/n) * sum (y_i - y_hat_i)^2
    + smooth, has one minimum, closed-form solution, easy gradient (Lab 03)
    + equals maximum likelihood when the noise is Gaussian
    - squares big errors: one outlier can dominate the whole cost

THE CHOICE OF LOSS DECIDES WHAT THE MODEL PREDICTS
    squared error  -> the MEAN of y given x
    absolute error -> the MEDIAN of y given x
    pinball loss q -> the q-th QUANTILE of y given x (asymmetric business costs)

LEVELS
    EASY   : Residuals, loss vs cost, and the MSE "bowl" over the model's parameters.
    MEDIUM : MSE vs MAE vs Huber: the best constant, outlier sensitivity.
    HARD   : Pick the loss from the business: asymmetric energy purchasing costs, and
             the log-target trap that under-budgets total healthcare spend.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, banner, make_energy_demand, make_house_sales, note, plt,
                          save_fig, show, step)

houses = make_house_sales(n=600, seed=RANDOM_STATE)
x = houses.sqft.to_numpy() / 1000                      # thousands of sqft -> readable parameters
y = houses.price.to_numpy() / 1000                     # $k

# %% =========================================================================
# LEVEL 1 (EASY): From residuals to a cost function
# ============================================================================
banner("Residuals, loss and cost", "EASY")

step("1.1  Three candidate lines, five houses")
cands = {"line A: 0 + 150*x": (0, 150), "line B: 20 + 160*x": (20, 160), "line C: 100 + 100*x": (100, 100)}
five = houses.head(5).assign(sqft_k=x[:5], price_k=y[:5])[["sqft_k", "price_k"]]
for name, (b0, b1) in cands.items():
    five[name.split(":")[0]] = b0 + b1 * five.sqft_k
print(five.round(1).to_string())

step("1.2  Loss per house -> cost for the whole dataset")
rows = {}
for name, (b0, b1) in cands.items():
    e = y - (b0 + b1 * x)
    rows[name] = {"mean residual": e.mean(), "MSE ($k^2)": np.mean(e ** 2), "RMSE ($k)": np.sqrt(np.mean(e ** 2)),
                  "MAE ($k)": np.mean(np.abs(e)), "worst error ($k)": np.abs(e).max()}
print(pd.DataFrame(rows).T.round(1).to_string())
note("""A residual is one number per house. A cost function squeezes all of them into
ONE number, so 'which line is best?' becomes 'which line has the lowest cost?'.
The mean residual is useless as a cost: positive and negative errors cancel.
Squaring (MSE) or taking absolute values (MAE) makes every miss count.""")

step("1.3  The MSE bowl: cost as a function of the parameters")
b0_grid = np.linspace(-150, 200, 141)
b1_grid = np.linspace(60, 240, 141)
B0, B1 = np.meshgrid(b0_grid, b1_grid)
J = np.mean((y[None, None, :] - (B0[..., None] + B1[..., None] * x[None, None, :])) ** 2, axis=-1)
i, j = np.unravel_index(np.argmin(J), J.shape)
ols = LinearRegression().fit(x[:, None], y)
show("Grid-search minimum (b0, b1)", f"({B0[i, j]:.1f}, {B1[i, j]:.1f})  MSE {J[i, j]:,.1f}")
show("Least squares solution (b0, b1)", f"({ols.intercept_:.1f}, {ols.coef_[0]:.1f})")
note("""Every point of the grid is a different model; its height is the cost. For
linear regression with MSE the surface is a smooth convex BOWL with a single
lowest point, so there are no local minima to get stuck in. Least squares jumps
straight to the bottom; gradient descent (Lab 03) walks down to it.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    cs = axes[0].contour(B0, B1, J, levels=np.geomspace(J.min() * 1.02, J.max(), 18), cmap="viridis")
    axes[0].plot(ols.intercept_, ols.coef_[0], "r*", ms=14, label="minimum (least squares)")
    for name, (b0, b1) in cands.items():
        axes[0].plot(b0, b1, "o", label=name)
    axes[0].set(xlabel="intercept b0 ($k)", ylabel="slope b1 ($k per 1,000 sqft)", title="MSE cost surface")
    axes[0].legend(fontsize=7)
    slope_cost = [np.mean((y - (ols.intercept_ + b * x)) ** 2) for b in b1_grid]
    axes[1].plot(b1_grid, slope_cost, color="#4c78a8")
    axes[1].axvline(ols.coef_[0], color="r", ls="--")
    axes[1].set(xlabel="slope b1 (intercept fixed at optimum)", ylabel="MSE", title="A slice through the bowl")
    save_fig(fig, "02_easy_mse_bowl")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Different losses, different answers
# ============================================================================
banner("MSE vs MAE vs Huber", "MEDIUM")

step("2.1  The best CONSTANT prediction depends on the loss")
rng = np.random.default_rng(RANDOM_STATE)
claims = np.round(rng.lognormal(np.log(3000), 1.1, 2000), -1)      # yearly medical claim per member ($)
consts = np.linspace(0, 15000, 3001)
best_mse = consts[np.argmin([np.mean((claims - c) ** 2) for c in consts])]
best_mae = consts[np.argmin([np.mean(np.abs(claims - c)) for c in consts])]
show("Mean of claims", f"${claims.mean():,.0f}")
show("Median of claims", f"${np.median(claims):,.0f}")
show("Constant minimising MSE", f"${best_mse:,.0f}")
show("Constant minimising MAE", f"${best_mae:,.0f}")
note("""Proof by search: the MSE-optimal constant is the MEAN, the MAE-optimal one is the
MEDIAN. On skewed money data they are very different. An insurer budgeting the
TOTAL claims bill needs the mean (total = mean x members). A product showing
'a typical member pays...' needs the median. Choosing the loss is choosing which
of these questions the model answers.""")

step("2.2  One data-entry error versus three loss functions")
clean_x, clean_y = x[:120], y[:120]
bad_x, bad_y = np.r_[clean_x, 4.6], np.r_[clean_y, 30.0]          # a ~$780k house typed as $30k
fits = {}
for name, mk in {"OLS (squared loss)": lambda: LinearRegression(),
                 "Huber (squared near 0, linear far away)": lambda: HuberRegressor(epsilon=1.35, max_iter=1000),
                 "LAD (absolute loss = median regression)": lambda: QuantileRegressor(quantile=0.5, alpha=0,
                                                                                    solver="highs")}.items():
    s_clean = mk().fit(clean_x[:, None], clean_y).coef_[0]
    s_bad = mk().fit(bad_x[:, None], bad_y).coef_[0]
    fits[name] = {"slope, clean data": s_clean, "slope, + 1 typo": s_bad, "change %": 100 * (s_bad / s_clean - 1)}
print(pd.DataFrame(fits).T.round(1).to_string())
note("""Squaring makes one $750k miss weigh as much as 2,500 misses of $15k, so OLS
tilts the whole line towards ONE bad row. Huber treats small errors like MSE and big ones like MAE;
absolute loss barely notices. But a robust loss only HIDES bad data: in Module 2
(Lab 03) we fixed the data instead. Use robust losses when big errors are real
but rare and you don't want them to dominate.""")

step("2.3  Why MSE is the default anyway")
note("""1) If the noise is Gaussian, minimising MSE = maximum likelihood: the 'most
probable' parameters. 2) It is differentiable everywhere, with gradient
2 * X^T (X b - y) / n: gradient descent and neural nets (Lab 10) rely on this.
3) It has a closed-form solution for linear models. 4) It predicts the mean,
which is what you need whenever predictions are summed (budgets, totals).""")

# %% =========================================================================
# LEVEL 3 (HARD): Choose the loss from the business cost
# ============================================================================
banner("The loss function is a business decision", "HARD")

step("3.1  Asymmetric costs: day-ahead energy purchasing")
# A utility buys tomorrow's electricity today at $60/MWh. If demand is higher than what it
# bought, the shortfall is bought on the spot market at $180/MWh. Unused energy is lost.
PRICE, SPOT = 60, 180
under_cost, over_cost = SPOT - PRICE, PRICE           # extra $ per MWh under- / over-bought
q_star = under_cost / (under_cost + over_cost)
train_e = make_energy_demand(n=600, seed=1)
test_e = make_energy_demand(n=3000, seed=2)
Xtr, Xte = train_e[["temp_c"]], test_e[["temp_c"]]


def poly(model):
    return make_pipeline(PolynomialFeatures(4), StandardScaler(), model)


def business_cost(y_true, bought):
    short = np.clip(y_true - bought, 0, None)
    over = np.clip(bought - y_true, 0, None)
    return float(np.mean(under_cost * short + over_cost * over))


models = {"OLS (MSE -> mean)": poly(LinearRegression()),
          "Median (MAE)": poly(QuantileRegressor(quantile=0.5, alpha=0, solver="highs")),
          f"Quantile q={q_star:.2f} (pinball)": poly(QuantileRegressor(quantile=q_star, alpha=0, solver="highs"))}
rows = {}
for name, m in models.items():
    p = m.fit(Xtr, train_e.demand_mwh).predict(Xte)
    yt = test_e.demand_mwh.to_numpy()
    rows[name] = {"RMSE (MWh)": np.sqrt(np.mean((yt - p) ** 2)), "days short %": 100 * np.mean(yt > p),
                  "extra cost $/day": business_cost(yt, p)}
res = pd.DataFrame(rows).T
print(res.round(1).to_string())
saving = res["extra cost $/day"].iloc[0] - res["extra cost $/day"].iloc[2]
show("Optimal quantile = under / (under + over)", f"{under_cost} / ({under_cost} + {over_cost}) = {q_star:.2f}")
show("Yearly saving vs the MSE model", f"${saving * 365:,.0f}")
note("""The MSE model has the BEST RMSE and the WORST business cost: it is short on half
the days, and every short MWh costs twice as much as a surplus one. The pinball
loss at q = 0.67 deliberately over-buys a little. Its RMSE is worse, the
business result is better. The metric on the dashboard must be the one the
business pays for.""")

step("3.2  The log-target trap: under-budgeting total healthcare spend")
rng = np.random.default_rng(5)
n = 6000
members = pd.DataFrame({"age": rng.integers(18, 65, n), "smoker": rng.random(n) < 0.2,
                        "chronic": rng.random(n) < 0.25})
log_mu = 7.2 + 0.03 * members.age + 1.1 * members.smoker + 0.8 * members.chronic
members["cost"] = np.exp(log_mu + rng.normal(0, 1.0, n))          # log-normal costs, sigma = 1
tr, te = members.iloc[:4000], members.iloc[4000:]
F = ["age", "smoker", "chronic"]
raw = LinearRegression().fit(tr[F], tr.cost)
logm = LinearRegression().fit(tr[F], np.log(tr.cost))
naive = np.exp(logm.predict(te[F]))
resid = np.log(tr.cost) - logm.predict(tr[F])
smear = np.mean(np.exp(resid))                                    # Duan's smearing estimator
budget = pd.DataFrame({
    "predicted total $M": [raw.predict(te[F]).sum() / 1e6, naive.sum() / 1e6, (naive * smear).sum() / 1e6],
    "actual total $M": te.cost.sum() / 1e6,
}, index=["OLS on dollars", "OLS on log(cost), exp(pred)", "log model x smearing factor"])
budget["error %"] = 100 * (budget["predicted total $M"] / budget["actual total $M"] - 1)
print(budget.round(2).to_string())
show("Smearing factor mean(exp(residual))", float(smear))
note("""Training on log(cost) is popular for skewed targets (it often improves R^2 and
MAE). But minimising squared error on the LOG scale predicts the mean of
log(cost); exp() of that is the GEOMETRIC mean, roughly the median, which is far
below the mean for skewed data. Sum those predictions into a budget and the
insurer is short by almost 40%. Fix: multiply by a smearing factor (or predict the
mean directly). Rule: know which statistic your loss estimates, and which one
the business decision needs.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    order = np.argsort(test_e.temp_c.to_numpy())
    axes[0].scatter(test_e.temp_c, test_e.demand_mwh, s=3, alpha=0.25, color="grey")
    for name, m in models.items():
        axes[0].plot(test_e.temp_c.to_numpy()[order], m.predict(Xte)[order], lw=2, label=name)
    axes[0].set(xlabel="temperature (C)", ylabel="demand (MWh)", title="Energy to buy: mean vs quantiles")
    axes[0].legend(fontsize=8)
    axes[1].barh(budget.index, budget["error %"], color=["#4c78a8", "#e45756", "#54a24b"])
    axes[1].axvline(0, color="k")
    axes[1].set(xlabel="error in total predicted spend (%)", title="Healthcare budget error")
    save_fig(fig, "02_hard_business_losses")

banner("KEY TAKEAWAYS")
print("""  1. Loss = penalty for one example; cost = average loss; training minimises the cost.
  2. MSE predicts the mean, MAE the median, pinball loss a quantile. Choose deliberately.
  3. MSE is smooth, convex and has a closed form, but one huge error can dominate it.
  4. Asymmetric business costs -> quantile q = cost_under / (cost_under + cost_over).
  5. A log-target model predicts the geometric mean: correct it before summing predictions.""")
