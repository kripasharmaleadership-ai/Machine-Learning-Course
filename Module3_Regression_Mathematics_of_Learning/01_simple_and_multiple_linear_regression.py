"""
LAB 01 - Simple and Multiple Linear Regression
==============================================

THE MODEL
    Simple   : y_hat = b0 + b1 * x                       (one feature -> a line)
    Multiple : y_hat = b0 + b1*x1 + b2*x2 + ... + bp*xp   (p features -> a hyperplane)

    b0 is the INTERCEPT (prediction when every feature is 0), b1..bp are the
    COEFFICIENTS (model parameters). "Training" = choosing the b's that make the
    predictions as close as possible to the real targets (Lab 02 defines "close").

READING A COEFFICIENT
    b_sqft = 160 means: "two houses that are IDENTICAL IN EVERY OTHER FEATURE, one
    1 sqft bigger, differ in predicted price by $160". The words "holding everything
    else fixed" are the whole difference between simple and multiple regression.

THE DATA
    Synthetic house sales whose TRUE pricing formula we know (ml_lab_utils.HOUSE_TRUTH),
    so we can check whether regression recovers it.

LEVELS
    EASY   : One feature: fit a line by hand (covariance / variance), then with sklearn.
    MEDIUM : Many features: the normal equation, "holding fixed", a coefficient that
             flips sign, categorical features, and non-linear relationships.
    HARD   : Coefficients are not importance: units, standardised coefficients,
             multicollinearity (unstable coefficients, stable predictions), bootstrap
             confidence intervals and the danger of extrapolation.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, HOUSE_FEATURES, HOUSE_TRUTH, NEIGHBORHOODS, RANDOM_STATE, banner,
                          make_house_sales, note, plt, save_fig, show, step)
from regression_toolkit import regression_metrics

houses = make_house_sales(n=2000, with_truth=True)
train, test = train_test_split(houses, test_size=0.25, random_state=RANDOM_STATE)
print(houses.head().to_string())

# %% =========================================================================
# LEVEL 1 (EASY): One feature, one line
# ============================================================================
banner("Simple linear regression: price ~ sqft", "EASY")
x, y = train.sqft.to_numpy(), train.price.to_numpy()

step("1.1  Fit the line by hand")
b1 = np.cov(x, y, ddof=0)[0, 1] / np.var(x)            # slope     = cov(x, y) / var(x)
b0 = y.mean() - b1 * x.mean()                           # intercept = the line passes through (x_bar, y_bar)
show("Slope b1 ($ per sqft)", f"{b1:,.1f}")
show("Intercept b0 ($)", f"{b0:,.0f}")
show("Correlation r(sqft, price)", float(np.corrcoef(x, y)[0, 1]))
note("""Closed form for ONE feature: slope = cov(x,y) / var(x). Equivalently,
slope = r * sd(y) / sd(x): correlation tells you HOW TIGHT the relationship is,
the slope tells you HOW STEEP it is, in the units of the problem.""")

step("1.2  The same with scikit-learn")
lr1 = LinearRegression().fit(train[["sqft"]], y)
show("sklearn slope", f"{lr1.coef_[0]:,.1f}")
show("sklearn intercept", f"{lr1.intercept_:,.0f}")
for size in [1200, 2000, 3000]:
    show(f"Predicted price of a {size:,} sqft house", f"${lr1.predict(pd.DataFrame({'sqft': [size]}))[0]:,.0f}")

m1 = regression_metrics(test.price, lr1.predict(test[["sqft"]]))
show("Test RMSE (sqft only)", f"${m1['RMSE']:,.0f}")
show("Test R^2  (sqft only)", m1["R2"])
note(f"""The fitted slope (~${b1:,.0f}/sqft) is close to the true value of a square foot
(${HOUSE_TRUTH['sqft']}) only by coincidence: big houses also have more bathrooms and sit
in better neighbourhoods (pushes the slope up) but have more bedrooms (pushes it
down, see 2.2). With one feature, the slope absorbs all of that. A simple
regression answers 'how does price move with size in the data?', not 'what is
a square foot worth?'. The intercept (${b0:,.0f} for a 0 sqft house) has no
real-world meaning: nobody sells a 0 sqft house. It just positions the line.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y / 1000, s=6, alpha=0.35, color="#4c78a8", label="house sales")
    grid = np.linspace(x.min(), x.max(), 50)
    ax.plot(grid, (b0 + b1 * grid) / 1000, color="#e45756", lw=2.5, label=f"y = {b0/1000:,.0f}k + {b1:,.0f} * sqft")
    i = np.argmax(np.abs(y - (b0 + b1 * x)))
    ax.vlines(x[i], (b0 + b1 * x[i]) / 1000, y[i] / 1000, color="k", ls=":", label="a residual")
    ax.set(xlabel="sqft", ylabel="price ($k)", title="Simple linear regression")
    ax.legend()
    save_fig(fig, "01_easy_simple_regression")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Many features at once
# ============================================================================
banner("Multiple linear regression: holding everything else fixed", "MEDIUM")

step("2.1  The normal equation: b = (X^T X)^-1 X^T y")
X = np.column_stack([np.ones(len(train)), train[HOUSE_FEATURES].to_numpy()])
beta = np.linalg.solve(X.T @ X, X.T @ y)                 # solve() instead of inverse(): more stable
lr = LinearRegression().fit(train[HOUSE_FEATURES], y)
comp = pd.DataFrame({"normal equation": beta, "sklearn": np.r_[lr.intercept_, lr.coef_],
                     "TRUE value": [HOUSE_TRUTH["intercept"]] + [HOUSE_TRUTH[f] for f in HOUSE_FEATURES]},
                    index=["intercept"] + HOUSE_FEATURES)
print(comp.round(0).to_string())
note("""Two ways, same answer: the normal equation is the exact minimiser of squared
error (Lab 02), and sklearn's LinearRegression solves the same least-squares
problem with a numerically safer method. The coefficients land close to the TRUE
values, except the intercept and distance, which still absorb the missing
neighbourhood effect (fixed in 2.3).""")

step("2.2  A coefficient that flips sign")
simple_bed = LinearRegression().fit(train[["bedrooms"]], y).coef_[0]
show("bedrooms alone      : $ per extra bedroom", f"{simple_bed:+,.0f}")
show("bedrooms + sqft + ...: $ per extra bedroom", f"{lr.coef_[HOUSE_FEATURES.index('bedrooms')]:+,.0f}")
show("TRUE effect (holding sqft fixed)", f"{HOUSE_TRUTH['bedrooms']:+,}")
show("corr(bedrooms, sqft)", float(train[["bedrooms", "sqft"]].corr().iloc[0, 1]))
note("""Alone, bedrooms looks very valuable because houses with more bedrooms are
BIGGER. Holding sqft fixed, one more bedroom means the same space cut into smaller
rooms, and buyers pay LESS. Both numbers are 'correct'; they answer different
questions. A seller asking 'should I add a wall to make a 4th bedroom?' needs the
multiple-regression coefficient.""")

step("2.3  Categorical features: one-hot encoding (drop one = the baseline)")
Xd = pd.get_dummies(train[HOUSE_FEATURES + ["neighborhood"]], columns=["neighborhood"], drop_first=True,
                    dtype=float)
lr_full = LinearRegression().fit(Xd, y)
hood = pd.DataFrame({"fitted premium vs Downtown":
                     pd.Series(lr_full.coef_, index=Xd.columns).filter(like="neighborhood_")})
hood.index = hood.index.str.replace("neighborhood_", "")
hood["TRUE premium vs Downtown"] = [NEIGHBORHOODS[h] - NEIGHBORHOODS["Downtown"] for h in hood.index]
print(hood.round(0).to_string())
Xd_test = pd.get_dummies(test[HOUSE_FEATURES + ["neighborhood"]], columns=["neighborhood"], drop_first=True,
                         dtype=float)[Xd.columns]
m_full = regression_metrics(test.price, lr_full.predict(Xd_test))
m_num = regression_metrics(test.price, lr.predict(test[HOUSE_FEATURES]))
show("Test RMSE: sqft only", f"${m1['RMSE']:,.0f}")
show("Test RMSE: 6 numeric features", f"${m_num['RMSE']:,.0f}")
show("Test RMSE: + neighbourhood", f"${m_full['RMSE']:,.0f}")
show("Irreducible noise in this world (sd)", "$25,000")
note("""Each neighbourhood coefficient is a premium RELATIVE TO THE DROPPED category
(Downtown, first alphabetically). With every true driver in the model, the test
RMSE approaches the noise level: the model has found the formula, and what remains
is luck (Lab 04: irreducible error).""")

step("2.4  Relationship between features and a continuous target: is it a line?")
rng = np.random.default_rng(1)
xs = rng.uniform(0, 10, 400)
shapes = {"linear": 3 * xs, "curved (quadratic)": (xs - 5) ** 2, "saturating (log)": 8 * np.log1p(xs),
          "no relation": np.zeros_like(xs)}
rows = {}
for name, f in shapes.items():
    yy = f + rng.normal(0, 2, len(xs))
    r_lin = LinearRegression().fit(xs[:, None], yy).score(xs[:, None], yy)
    Xq = np.column_stack([xs, xs ** 2])
    r_quad = LinearRegression().fit(Xq, yy).score(Xq, yy)
    rows[name] = {"pearson r": np.corrcoef(xs, yy)[0, 1], "R2 line": r_lin, "R2 with x^2 term": r_quad}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""A U-shaped relationship has correlation ~0 and a straight line explains
nothing, yet x is extremely predictive. 'Linear' regression is linear in the
COEFFICIENTS, not in x: add x^2 (or log x) as a feature and the same algorithm
fits curves. Always LOOK at feature-vs-target plots before trusting r.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
    for ax, (name, f) in zip(axes, shapes.items()):
        ax.scatter(xs, f + rng.normal(0, 2, len(xs)), s=4, alpha=0.5)
        ax.set_title(f"{name}\nr = {rows[name]['pearson r']:.2f}", fontsize=9)
    save_fig(fig, "01_medium_relationship_shapes")

# %% =========================================================================
# LEVEL 3 (HARD): Why coefficients are not "importance"
# ============================================================================
banner("Coefficients: units, collinearity and uncertainty", "HARD")

step("3.1  Coefficient size depends on units")
alt = train.assign(sqft_thousands=train.sqft / 1000, distance_m=train.distance_km * 1000)
c_units = LinearRegression().fit(alt[["sqft_thousands", "age", "distance_m"]], y).coef_
show("sqft in thousands: coefficient", f"{c_units[0]:,.0f}")
show("distance in metres: coefficient", f"{c_units[2]:,.2f}")
std_lr = LinearRegression().fit(StandardScaler().fit_transform(train[HOUSE_FEATURES]), y)
print(pd.Series(std_lr.coef_, index=HOUSE_FEATURES, name="$ per 1 std dev").sort_values(key=abs,
                                                                                         ascending=False)
      .round(0).to_string())
note("""Measure distance in metres and its coefficient becomes 1000x smaller: the
feature did not become less important. To compare features, standardise them
first: the coefficient is then '$ per one standard deviation'. Even that measures
the effect of a TYPICAL change, not causal importance (Module 1, Lab 12).""")

step("3.2  Multicollinearity: two features that say the same thing")
# The listing site ALSO stores the floor area measured by a second appraiser in square metres
coll = train.assign(sqm_appraiser=train.sqft / 10.764 + np.random.default_rng(3).normal(0, 4, len(train)))
feats_c = HOUSE_FEATURES + ["sqm_appraiser"]
show("corr(sqft, sqm_appraiser)", float(coll[["sqft", "sqm_appraiser"]].corr().iloc[0, 1]))
boot = []
rng = np.random.default_rng(7)
q = coll[feats_c].iloc[[0]]
for _ in range(300):
    idx = rng.integers(0, len(coll), len(coll))
    m = LinearRegression().fit(coll[feats_c].iloc[idx], coll.price.iloc[idx])
    boot.append({"b_sqft": m.coef_[0], "b_sqm": m.coef_[-1],
                 "sqft effect total ($/sqft)": m.coef_[0] + m.coef_[-1] / 10.764,
                 "prediction house #0": m.predict(q)[0]})
boot = pd.DataFrame(boot)
summary = boot.describe(percentiles=[.025, .975]).T[["mean", "2.5%", "97.5%"]]
print(summary.round(1).to_string())
note("""With an almost-duplicate feature, the data cannot decide how to SPLIT the
size effect between the two columns: each coefficient swings wildly between
bootstrap samples, sometimes negative. Their combined effect and the predictions
stay stable. Lesson: under collinearity, never interpret single coefficients
(and never use them to justify a business decision). Remedies: drop one,
combine them, or use Ridge regularisation (Lab 05).""")

step("3.3  Bootstrap confidence intervals for the honest model")
rng = np.random.default_rng(11)
bs = np.array([LinearRegression().fit(Xd.iloc[i], y[i]).coef_[:len(HOUSE_FEATURES)]
               for i in (rng.integers(0, len(Xd), len(Xd)) for _ in range(300))])
ci = pd.DataFrame({"estimate": lr_full.coef_[:len(HOUSE_FEATURES)], "95% low": np.percentile(bs, 2.5, 0),
                   "95% high": np.percentile(bs, 97.5, 0),
                   "TRUE": [HOUSE_TRUTH[f] for f in HOUSE_FEATURES]}, index=HOUSE_FEATURES)
ci["truth inside CI"] = (ci["95% low"] <= ci.TRUE) & (ci.TRUE <= ci["95% high"])
print(ci.round(0).to_string())
note("""A coefficient is an ESTIMATE from one sample. Bootstrapping (re-fitting on
resampled rows) shows how much it would move with different data. With 1,500
houses and the right features, every interval contains the true value.""")

step("3.4  Extrapolation: the line goes on forever, reality does not")
mansion = pd.DataFrame([{"sqft": 15000, "bedrooms": 4, "bathrooms": 3, "age": 10, "garage_spaces": 2,
                         "distance_km": 3}])
tiny = pd.DataFrame([{"sqft": 150, "bedrooms": 1, "bathrooms": 1, "age": 79, "garage_spaces": 0,
                      "distance_km": 35}])
show("Training sqft range", f"{train.sqft.min():,.0f} - {train.sqft.max():,.0f}")
show("Predicted price: 15,000 sqft mansion", f"${lr.predict(mansion)[0]:,.0f}")
show("Predicted price: 150 sqft shed, 35 km out", f"${lr.predict(tiny)[0]:,.0f}")
note("""Inside the training range, the model is well calibrated. Outside it, a
linear model keeps extending the line: the mansion gets an exact-looking price
from a formula never tested on mansions, and the shed gets a NEGATIVE price.
Production rule: check that inputs fall inside the training range, and flag
out-of-range requests for a human.""")

banner("KEY TAKEAWAYS")
print("""  1. Simple regression: slope = cov/var = r * sd(y)/sd(x). It mixes in every omitted factor.
  2. Multiple regression coefficients mean 'holding the other features fixed' - signs can flip.
  3. Linear in the COEFFICIENTS: add x^2, log(x), one-hot columns and it fits curves and categories.
  4. Coefficient size depends on units; standardise to compare; it is still not causal importance.
  5. Collinear features -> unstable coefficients, stable predictions. Don't interpret them; fix them.
  6. Never trust a linear model outside the range of its training data.""")
