"""
LAB 08 - Regression Evaluation Metrics: MAE, MSE, RMSE, R^2, Adjusted R^2
=========================================================================

THE METRICS (e_i = y_i - y_hat_i, n rows, p features)
    MAE      = mean(|e_i|)                    "typical miss", in the target's units ($)
    MSE      = mean(e_i^2)                    the training loss; units are $^2 (hard to read)
    RMSE     = sqrt(MSE)                      in $, but big misses weigh more than in MAE
    R^2      = 1 - SS_res / SS_tot            share of the target's variance explained
               SS_res = sum e_i^2, SS_tot = sum (y_i - mean(y))^2
               1 = perfect, 0 = no better than predicting the mean, < 0 = WORSE than the mean
    Adj R^2  = 1 - (1 - R^2)(n - 1)/(n - p - 1)
               penalises features that don't pull their weight

WHICH ONE?
    MAE  : costs grow linearly with the miss (most pricing / staffing questions)
    RMSE : big misses are disproportionately bad (safety stock, credit exposure)
    R^2  : comparing models ON THE SAME DATA; not comparable across datasets
    Always report a baseline next to them, and the metric the business pays for.

LEVELS
    EASY   : Compute every metric by hand for 8 houses and read each one in plain words.
    MEDIUM : When metrics disagree: MAE vs RMSE, negative R^2, R^2 depends on the population.
    HARD   : Adjusted R^2 vs junk features; "is model B really better?" (bootstrap);
             and the segment the average hides.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ml_lab_utils import HAS_MPL, HOUSE_FEATURES, RANDOM_STATE, banner, make_house_sales, note, plt, save_fig, show, step
from regression_toolkit import adjusted_r2, bootstrap_compare, regression_metrics, residual_table

houses = make_house_sales(n=3000, seed=31)
tr, te = train_test_split(houses, test_size=0.3, random_state=RANDOM_STATE)
Xtr = pd.get_dummies(tr[HOUSE_FEATURES + ["neighborhood"]], columns=["neighborhood"], drop_first=True, dtype=float)
Xte = pd.get_dummies(te[HOUSE_FEATURES + ["neighborhood"]], columns=["neighborhood"], drop_first=True,
                     dtype=float)[Xtr.columns]
model = LinearRegression().fit(Xtr, tr.price)
pred = model.predict(Xte)

# %% =========================================================================
# LEVEL 1 (EASY): Every metric by hand
# ============================================================================
banner("Five metrics, computed by hand", "EASY")
eight = te.head(8).assign(predicted=pred[:8].round(-2))[["sqft", "neighborhood", "price", "predicted"]]
eight["error"] = eight.price - eight.predicted
eight["|error|"] = eight.error.abs()
eight["error^2"] = eight.error ** 2
print(eight.to_string(float_format=lambda v: f"{v:,.0f}"))
y8, p8 = eight.price.to_numpy(), eight.predicted.to_numpy()
mae = np.mean(np.abs(y8 - p8))
mse = np.mean((y8 - p8) ** 2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum((y8 - p8) ** 2) / np.sum((y8 - y8.mean()) ** 2)
adj = adjusted_r2(r2, n=8, p=3)
step("By hand")
show("MAE  = mean |error|", f"${mae:,.0f}")
show("MSE  = mean error^2", f"{mse:,.0f} $^2")
show("RMSE = sqrt(MSE)", f"${rmse:,.0f}")
show("R^2  = 1 - SS_res / SS_tot", f"{r2:.3f}")
show("Adj R^2 (n=8, pretend p=3)", f"{adj:.3f}")
step("scikit-learn agrees")
show("mean_absolute_error / root_mean_squared_error", f"${mean_absolute_error(y8, p8):,.0f} / "
                                                     f"${root_mean_squared_error(y8, p8):,.0f}")
show("mean_squared_error / r2_score", f"{mean_squared_error(y8, p8):,.0f} / {r2_score(y8, p8):.3f}")

step("On the full test set, read in plain words")
m = regression_metrics(te.price, pred, n_features=Xtr.shape[1])
base = regression_metrics(te.price, np.full(len(te), tr.price.mean()))
print(pd.DataFrame({"linear model": m, "baseline: predict the mean": base}).to_string(
    float_format=lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"))
note(f"""MAE ${m['MAE']:,.0f}: 'on average our valuation is off by about ${m['MAE'] / 1000:.0f}k.' RMSE
${m['RMSE']:,.0f}: always >= MAE; the gap grows when there are some big misses. R^2
{m['R2']:.2f}: 'the model explains {100 * m['R2']:.0f}% of the variation in prices'; the mean
baseline scores 0 by construction. MSE is the training loss but in $^2:
nobody can read '6e8 dollars squared', so report RMSE instead.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): When metrics disagree
# ============================================================================
banner("Metrics answer different questions", "MEDIUM")

step("2.1  Same MAE, very different RMSE")
rng = np.random.default_rng(0)
y = te.price.to_numpy()
steady = y + rng.choice([-1, 1], len(y)) * 20_000                         # always off by exactly $20k
spiky = y.copy()
big = rng.random(len(y)) < 0.1
spiky[big] += rng.choice([-1, 1], big.sum()) * 200_000                     # perfect 90%, off by $200k 10%
print(pd.DataFrame({"model A (always $20k off)": regression_metrics(y, steady),
                    "model B (perfect, but 10% are $200k off)": regression_metrics(y, spiky)}).T[
    ["MAE", "RMSE", "MedianAE", "R2"]].round(3).to_string())
note("""Practically the same MAE (~$20k). Model B's RMSE is 3x larger, and its median error is 0.
Which is better depends on the business: for an instant-offer home buyer, a
$200k overpayment on 1 house in 10 can wipe out the margin of the other 9, so
RMSE (or a tail metric) is the right lens. For a 'price estimate' widget on a
listing site, model B is perfect for 90% of users. Report BOTH, plus the
error distribution.""")

step("2.2  R^2 can be negative")
for name, p in {"the model": pred, "predict the training mean": np.full(len(te), tr.price.mean()),
                "model + a $60k bug in production": pred + 60_000,
                "a model trained on another city (x0.6)": pred * 0.6}.items():
    show(f"R^2  {name}", r2_score(te.price, p))
note("""R^2 = 0 is the 'predict the average' baseline. A model that is worse than that,
such as a systematic offset after a pipeline bug or a model reused on a
different market, scores BELOW zero. It isn't a mistake: it's a strong alarm.""")

step("2.3  R^2 depends on the population, not only on the model")
bands = {}
for label, lo, hi in [("all houses", 0, np.inf), ("1,500-2,200 sqft only", 1500, 2200),
                      ("1,700-1,900 sqft only", 1700, 1900)]:
    mk = ((te.sqft >= lo) & (te.sqft < hi)).to_numpy()
    bands[label] = {"n": int(mk.sum()), "RMSE $": root_mean_squared_error(te.price[mk], pred[mk]),
                    "R^2": r2_score(te.price[mk], pred[mk]), "price sd $": te.price[mk].std()}
print(pd.DataFrame(bands).T.to_string(float_format=lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"))
note("""Same model, similar or even SMALLER errors, yet R^2 drops from 0.96 to 0.86 when
we evaluate only on similar-sized houses. R^2 divides by the target's
variance: a population with a narrower spread of prices makes the same errors
look bigger. Never compare R^2 across different datasets, segments or
time periods. Compare RMSE / MAE in business units instead.""")

# %% =========================================================================
# LEVEL 3 (HARD): Adjusted R^2, real differences, hidden segments
# ============================================================================
banner("Using metrics to make decisions", "HARD")

step("3.1  Adjusted R^2 vs junk features (120 training houses)")
small_tr = tr.sample(120, random_state=3)
rng = np.random.default_rng(1)
rows = {}
for k in [0, 5, 20, 50, 80, 100]:
    junk_tr = rng.normal(0, 1, (len(small_tr), k))
    junk_te = rng.normal(0, 1, (len(te), k))
    A = np.column_stack([small_tr[HOUSE_FEATURES].to_numpy(), junk_tr])
    B = np.column_stack([te[HOUSE_FEATURES].to_numpy(), junk_te])
    mdl = LinearRegression().fit(A, small_tr.price)
    r2_tr = mdl.score(A, small_tr.price)
    rows[f"6 real + {k} junk"] = {"features p": A.shape[1], "train R^2": r2_tr,
                                  "train adjusted R^2": adjusted_r2(r2_tr, len(small_tr), A.shape[1]),
                                  "test R^2": mdl.score(B, te.price)}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""Plain R^2 on the training data can only go UP when you add a feature, even pure
noise: with 100 junk columns it approaches 1. Adjusted R^2 charges each feature
a price (n - 1)/(n - p - 1) and FALLS as junk accumulates, tracking the test
R^2 much better. It is a quick in-sample sanity check; the real arbiter is still
held-out data (when p approaches n, adjusted R^2 itself becomes unreliable).""")

step("3.2  Is model B really better than model A? (bootstrap the test set)")
tree = DecisionTreeRegressor(min_samples_leaf=10, random_state=0).fit(Xtr, tr.price)
lin_minus = LinearRegression().fit(Xtr.drop(columns=["bathrooms"]), tr.price)
comparisons = {
    "A=full linear  B=tree": (pred, tree.predict(Xte)),
    "A=full linear  B=linear without bathrooms": (pred, lin_minus.predict(Xte.drop(columns=["bathrooms"]))),
}
small_te_idx = np.random.default_rng(4).choice(len(te), 60, replace=False)
rows = {}
for name, (pa, pb) in comparisons.items():
    for label, idx in [("full test (900 rows)", np.arange(len(te))), ("small test (60 rows)", small_te_idx)]:
        res = bootstrap_compare(te.price.to_numpy()[idx], pa[idx], pb[idx], n_boot=2000)
        rows[f"{name} | {label}"] = {"RMSE(B) - RMSE(A)": res["diff (B - A)"], "95% CI low": res["ci_low"],
                                     "95% CI high": res["ci_high"],
                                     "verdict": "A better" if res["ci_low"] > 0 else
                                     ("B better" if res["ci_high"] < 0 else "can't tell")}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:,.0f}"))
note("""A single RMSE number has sampling noise. Bootstrapping the test rows gives a
confidence interval for the DIFFERENCE between two models. The tree is worse
on any test set. Dropping bathrooms (a real but small effect) is measurably
worse on 900 test houses, yet on a 60-house test set the interval includes 0:
the data cannot tell. Before announcing 'model B wins by 2%' (on a
leaderboard, in a slide), check that the interval excludes zero.""")

step("3.3  The segment the average hides")
luxury = houses.copy()
rng = np.random.default_rng(8)
lux = (luxury.neighborhood == "Lakeside") & (luxury.sqft > 2600)
luxury.loc[lux, "price"] *= 1.25                                     # waterfront premium the model can't see
l_tr, l_te = train_test_split(luxury, test_size=0.3, random_state=RANDOM_STATE)
Lx_tr = pd.get_dummies(l_tr[HOUSE_FEATURES + ["neighborhood"]], columns=["neighborhood"], drop_first=True, dtype=float)
Lx_te = pd.get_dummies(l_te[HOUSE_FEATURES + ["neighborhood"]], columns=["neighborhood"], drop_first=True,
                       dtype=float)[Lx_tr.columns]
lp = LinearRegression().fit(Lx_tr, l_tr.price).predict(Lx_te)
seg = np.where((l_te.neighborhood == "Lakeside") & (l_te.sqft > 2600), "Lakeside > 2,600 sqft",
               np.where(l_te.sqft > 2600, "other > 2,600 sqft", "<= 2,600 sqft"))
overall = regression_metrics(l_te.price, lp)
show("Overall test R^2 / MAE", f"{overall['R2']:.3f} / ${overall['MAE']:,.0f}")
print(residual_table(l_te.price, lp, pd.Series(seg)).round(0).to_string())
note("""The overall numbers look excellent. Per segment, the large Lakeside homes are
UNDER-valued by six figures on average (mean residual > 0): the model has no
feature for the waterfront premium. These are under 3% of houses but a large share
of the dollar exposure. A metric report without a per-segment table and a
mean-residual (bias) column is incomplete.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    res_d = pd.DataFrame(rows).T
    axes[0].plot(list(range(len(res_d))), res_d["RMSE(B) - RMSE(A)"], "o", color="k")
    axes[0].vlines(range(len(res_d)), res_d["95% CI low"].astype(float), res_d["95% CI high"].astype(float))
    axes[0].axhline(0, color="r", ls="--")
    axes[0].set_xticks(range(len(res_d)), [s.replace(" | ", "\n") for s in res_d.index], fontsize=6)
    axes[0].set(ylabel="RMSE(B) - RMSE(A) ($)", title="Bootstrap 95% CI of the difference")
    for s in np.unique(seg):
        mk = seg == s
        axes[1].scatter(lp[mk] / 1000, (l_te.price.to_numpy()[mk] - lp[mk]) / 1000, s=6, alpha=0.5, label=s)
    axes[1].axhline(0, color="k")
    axes[1].set(xlabel="predicted price ($k)", ylabel="residual ($k)", title="Residuals by segment")
    axes[1].legend(fontsize=7)
    save_fig(fig, "08_hard_metric_decisions")

banner("KEY TAKEAWAYS")
print("""  1. MAE = typical miss; RMSE punishes big misses; MSE is the loss, not a report metric.
  2. R^2 = share of variance explained vs the mean baseline; it can be negative.
  3. R^2 depends on the population's spread: never compare it across datasets or segments.
  4. Adjusted R^2 penalises useless features; held-out data is still the final judge.
  5. Report a CI for model differences and a per-segment table with mean residuals.""")
