"""
LAB 07 - Decision-Tree Regression
=================================

HOW A REGRESSION TREE WORKS
    1. Try every feature and every split point "x_j <= t".
    2. Keep the split that reduces the squared error (MSE) the most: each side
       predicts the MEAN of its training rows.
    3. Repeat inside each side until a stopping rule (max_depth, min_samples_leaf...).
    The prediction for a new row = the average target of the leaf it falls into.

WHAT THAT IMPLIES
    + No scaling, no linearity assumption; finds thresholds and INTERACTIONS by itself
    + Readable when shallow: a tree of depth 3 is a set of if/else business rules
    - Piecewise CONSTANT: a step function, not a smooth curve
    - Cannot EXTRAPOLATE: it never predicts outside the range of training targets
    - High variance: small data changes -> a different tree (-> ensembles, later module)

LEVELS
    EASY   : A step function on house prices; print the tree as rules.
    MEDIUM : No scaling needed; interactions for free; tuning depth/leaf size;
             cost-complexity pruning.
    HARD   : Four production failure modes: extrapolating a market trend, instability,
             non-monotonic prices, and misleading feature importance.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

from ml_lab_utils import (HAS_MPL, HOUSE_FEATURES, RANDOM_STATE, banner, make_house_sales, note, plt, save_fig,
                          show, step)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


houses = make_house_sales(n=2500, seed=21)
tr, te = train_test_split(houses, test_size=0.3, random_state=RANDOM_STATE)
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): A tree is a step function
# ============================================================================
banner("Regression trees predict leaf averages", "EASY")

step("1.1  One feature: price ~ sqft")
rows, fits = {}, {}
for d in [1, 2, 3, 6, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=0).fit(tr[["sqft"]], tr.price)
    fits[d] = t
    rows[f"max_depth={d}"] = {"leaves": t.get_n_leaves(), "distinct predictions": len(np.unique(t.predict(te[["sqft"]]))),
                              "train RMSE": rmse(tr.price, t.predict(tr[["sqft"]])),
                              "test RMSE": rmse(te.price, t.predict(te[["sqft"]]))}
line = LinearRegression().fit(tr[["sqft"]], tr.price)
rows["linear regression"] = {"leaves": "-", "distinct predictions": "continuous",
                             "train RMSE": rmse(tr.price, line.predict(tr[["sqft"]])),
                             "test RMSE": rmse(te.price, line.predict(te[["sqft"]]))}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:,.0f}"))
note("""A depth-1 tree (a 'stump') asks ONE question and answers with one of two
prices. Each extra level doubles the number of possible answers. An unlimited
tree grows ~300 leaves: training error keeps falling while test error rises
(overfitting, Lab 04). When the truth is a straight line, a line
beats any staircase.""")

step("1.2  A depth-2 tree on all features, printed as business rules")
t2 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(tr[HOUSE_FEATURES], tr.price)
print(export_text(t2, feature_names=HOUSE_FEATURES, decimals=0))
note("""Each path from the top to a leaf is a rule an appraiser can read: 'IF sqft <=
X AND ... THEN price = $Y (the average of those training houses)'. This
readability is the main business reason to choose a SHALLOW tree.""")

if HAS_MPL:
    grid = pd.DataFrame({"sqft": np.linspace(tr.sqft.min(), tr.sqft.max(), 600)})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(tr.sqft, tr.price / 1000, s=3, alpha=0.2, color="grey")
    for d in [1, 3, None]:
        ax.plot(grid.sqft, fits[d].predict(grid) / 1000, lw=1.5 if d else 0.6, label=f"tree max_depth={d}")
    ax.plot(grid.sqft, line.predict(grid) / 1000, "k--", lw=2, label="linear regression")
    ax.set(xlabel="sqft", ylabel="price ($k)", title="A regression tree is a step function")
    ax.legend(fontsize=8)
    save_fig(fig, "07_easy_step_function")

# %% =========================================================================
# LEVEL 2 (MEDIUM): What trees are good at, and how to tune them
# ============================================================================
banner("Strengths, tuning and pruning", "MEDIUM")

step("2.1  No feature scaling needed")
sc = StandardScaler().fit(tr[HOUSE_FEATURES])
a = DecisionTreeRegressor(max_depth=6, random_state=0).fit(tr[HOUSE_FEATURES], tr.price).predict(te[HOUSE_FEATURES])
b = DecisionTreeRegressor(max_depth=6, random_state=0).fit(sc.transform(tr[HOUSE_FEATURES]), tr.price).predict(
    sc.transform(te[HOUSE_FEATURES]))
show("Max |prediction difference| raw vs scaled", f"${np.abs(a - b).max():,.2f}")
note("""A split 'sqft <= 1,850' and 'z_sqft <= 0.31' select exactly the same houses.
Trees only care about the ORDER of values, not their scale.""")

step("2.2  Interactions and thresholds for free: yearly insurance cost")
rng = np.random.default_rng(4)
n = 3000
ins = pd.DataFrame({"age": rng.integers(18, 65, n), "bmi": np.clip(rng.normal(29, 6, n), 16, 50),
                    "smoker": (rng.random(n) < 0.2).astype(int)})
ins["cost"] = (2500 + 260 * ins.age + 21000 * ins.smoker * (ins.bmi >= 30) + 9000 * ins.smoker
               + rng.normal(0, 2500, n))
F2 = ["age", "bmi", "smoker"]
i_tr, i_te = ins.iloc[:2000], ins.iloc[2000:]
lin = LinearRegression().fit(i_tr[F2], i_tr.cost)
tree = GridSearchCV(DecisionTreeRegressor(random_state=0), {"max_depth": [2, 3, 4, 5, 6, 8],
                                                            "min_samples_leaf": [5, 20, 50]}, cv=cv).fit(i_tr[F2], i_tr.cost)
lin_x = LinearRegression().fit(i_tr[F2].assign(smoker_obese=i_tr.smoker * (i_tr.bmi >= 30)), i_tr.cost)
print(pd.DataFrame({
    "linear (age, bmi, smoker)": {"test RMSE": rmse(i_te.cost, lin.predict(i_te[F2]))},
    f"tree {tree.best_params_}": {"test RMSE": rmse(i_te.cost, tree.predict(i_te[F2]))},
    "linear + smoker x (bmi>=30) feature": {"test RMSE": rmse(i_te.cost, lin_x.predict(
        i_te[F2].assign(smoker_obese=i_te.smoker * (i_te.bmi >= 30))))},
}).T.round(0).to_string())
note("""The truth contains a THRESHOLD (BMI 30) inside an INTERACTION (only for smokers).
A linear model can't represent either unless someone engineers the feature.
The tree discovers 'smoker? then bmi >= 30?' on its own. A linear model given
the right feature is even better here, because the age effect really is a
straight line, which a tree must approximate with steps. You will meet exactly
this pattern on real data in Lab 09.""")

step("2.3  Tuning: max_depth x min_samples_leaf (5-fold CV on house prices)")
grid = GridSearchCV(DecisionTreeRegressor(random_state=0),
                    {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 20, 50]},
                    cv=cv, scoring="neg_root_mean_squared_error").fit(tr[HOUSE_FEATURES], tr.price)
tab = pd.DataFrame(grid.cv_results_)[["param_max_depth", "param_min_samples_leaf", "mean_test_score"]]
tab["CV RMSE"] = -tab.pop("mean_test_score")
print(tab.pivot(index="param_max_depth", columns="param_min_samples_leaf", values="CV RMSE").round(0).to_string())
show("Best", f"{grid.best_params_}  CV RMSE ${-grid.best_score_:,.0f}")
note("""max_depth limits how many questions are asked; min_samples_leaf says every answer
must be an average over at least k houses. The second is usually the better
guard: it stops tiny, noisy leaves wherever they would form, while still
letting the tree go deep where there is plenty of data.""")

step("2.4  Cost-complexity pruning: grow big, then cut back")
full = DecisionTreeRegressor(random_state=0, min_samples_leaf=5).fit(tr[HOUSE_FEATURES], tr.price)
path = full.cost_complexity_pruning_path(tr[HOUSE_FEATURES], tr.price)
ccps = np.unique(np.quantile(path.ccp_alphas, np.linspace(0, 0.995, 25)))
scores = [-cross_val_score(DecisionTreeRegressor(random_state=0, min_samples_leaf=5, ccp_alpha=a),
                           tr[HOUSE_FEATURES], tr.price, cv=cv, scoring="neg_root_mean_squared_error").mean()
          for a in ccps]
best_ccp = ccps[int(np.argmin(scores))]
pruned = DecisionTreeRegressor(random_state=0, min_samples_leaf=5, ccp_alpha=best_ccp).fit(tr[HOUSE_FEATURES], tr.price)
show("Leaves before / after pruning", f"{full.get_n_leaves()} / {pruned.get_n_leaves()}")
show("Test RMSE before / after pruning", f"${rmse(te.price, full.predict(te[HOUSE_FEATURES])):,.0f} / "
                                         f"${rmse(te.price, pruned.predict(te[HOUSE_FEATURES])):,.0f}")
lin_h = LinearRegression().fit(tr[HOUSE_FEATURES], tr.price)
show("Linear regression test RMSE (same features)", f"${rmse(te.price, lin_h.predict(te[HOUSE_FEATURES])):,.0f}")
note("""Pruning is regularisation for trees: cost = MSE + ccp_alpha * (number of leaves).
It removes the branches whose error reduction does not pay for their added
complexity. Even tuned, the tree loses to linear regression here, because this
house market really IS additive and linear. Trees win when the truth has
thresholds and interactions (2.2).""")

# %% =========================================================================
# LEVEL 3 (HARD): Four ways a regression tree fails in production
# ============================================================================
banner("Production failure modes", "HARD")

step("3.1  Trees cannot extrapolate: a rising market")
rng = np.random.default_rng(9)
hs = make_house_sales(n=4000, seed=33)
hs["month"] = rng.integers(0, 72, len(hs))                         # 6 years of sales
hs["price"] = (hs.price * 1.006 ** hs.month).round(-2)              # prices rise 0.6% per month
F3 = HOUSE_FEATURES + ["month"]
past, future = hs[hs.month < 60], hs[hs.month >= 60]                # train on 5 years, predict the 6th
t_m = DecisionTreeRegressor(min_samples_leaf=20, random_state=0).fit(past[F3], past.price)
l_m = LinearRegression().fit(past[F3], past.price)
rows = {}
for q in range(4):
    blk = future[(future.month >= 60 + 3 * q) & (future.month < 63 + 3 * q)]
    rows[f"quarter {q + 1}"] = {"tree mean error $": (blk.price - t_m.predict(blk[F3])).mean(),
                                "linear mean error $": (blk.price - l_m.predict(blk[F3])).mean(),
                                "tree RMSE": rmse(blk.price, t_m.predict(blk[F3])),
                                "linear RMSE": rmse(blk.price, l_m.predict(blk[F3]))}
print(pd.DataFrame(rows).T.round(0).to_string())
note("""The tree's last split on 'month' puts every future sale into the 'latest
months' leaf: it predicts last year's prices forever. Its under-prediction
(mean error > 0) GROWS every quarter. The linear model extends the trend. A tree
never predicts a value outside the range of the training targets: dangerous for
anything with a trend (prices, demand, inflation). Fix: model the trend
separately, predict ratios or differences, or retrain often.""")

step("3.2  Instability: resample 300 houses, get a different tree")


def top_structure(t, max_depth=2):
    """The features used in the top 7 nodes (depth 0-2), read level by level."""
    out, queue = [], [(0, 0)]
    while queue:
        node, d = queue.pop(0)
        if d > max_depth or t.tree_.feature[node] < 0:
            continue
        out.append(HOUSE_FEATURES[t.tree_.feature[node]])
        queue += [(t.tree_.children_left[node], d + 1), (t.tree_.children_right[node], d + 1)]
    return " | ".join(out)


small = tr.sample(300, random_state=1)
structures, root_thr = [], []
for k in range(200):
    s = small.sample(len(small), replace=True, random_state=k)
    t = DecisionTreeRegressor(max_depth=4, random_state=0).fit(s[HOUSE_FEATURES], s.price)
    structures.append(top_structure(t))
    root_thr.append(t.tree_.threshold[0])
show("Distinct top-7-node structures in 200 trees", pd.Series(structures).nunique())
show("Most common structure (share of trees)", f"{pd.Series(structures).value_counts(normalize=True).iloc[0]:.0%}")
show("Root split 'sqft <= t': 90% range of t", f"{np.percentile(root_thr, 5):,.0f} - {np.percentile(root_thr, 95):,.0f}")
preds = np.array([DecisionTreeRegressor(max_depth=4, random_state=0).fit(
    s[HOUSE_FEATURES], s.price).predict(te[HOUSE_FEATURES].iloc[:200])
    for s in (small.sample(len(small), replace=True, random_state=k) for k in range(50))])
lin_preds = np.array([LinearRegression().fit(s[HOUSE_FEATURES], s.price).predict(te[HOUSE_FEATURES].iloc[:200])
                      for s in (small.sample(len(small), replace=True, random_state=k) for k in range(50))])
show("Avg sd of a house's predicted price: tree", f"${preds.std(0).mean():,.0f}")
show("Avg sd of a house's predicted price: linear", f"${lin_preds.std(0).mean():,.0f}")
note("""With 300 houses, every resample gives different rules: the root threshold moves
by hundreds of sqft, and the questions asked below it change in most trees. A single house's predicted price moves by
much more than with a linear model. The printed 'business rules' of 1.2 are
therefore fragile. Averaging many trees (random forests, gradient boosting) is
the standard cure, covered in the next module.""")

step("3.3  Monotonicity: can a BIGGER house be cheaper?")


def monotonic_violations(model, X, feature="sqft", bump=100):
    up = X.copy()
    up[feature] = up[feature] + bump
    return float(np.mean(model.predict(up) < model.predict(X) - 1))


t_free = DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(tr[HOUSE_FEATURES], tr.price)
cst = [1 if f == "sqft" else 0 for f in HOUSE_FEATURES]
t_mono = DecisionTreeRegressor(min_samples_leaf=5, random_state=0, monotonic_cst=cst).fit(tr[HOUSE_FEATURES], tr.price)
for name, m in [("unconstrained tree", t_free), ("tree with monotonic_cst on sqft", t_mono)]:
    show(f"{name}: % houses cheaper after +100 sqft", f"{100 * monotonic_violations(m, te[HOUSE_FEATURES]):.1f}%")
    show(f"{name}: test RMSE", f"${rmse(te.price, m.predict(te[HOUSE_FEATURES])):,.0f}")
note("""In the unconstrained tree, adding 100 sqft LOWERS the price for a noticeable
share of houses: noise in neighbouring leaves. A customer who sees that will
not trust the product, and a regulator may not accept it. monotonic_cst forces
'more sqft never lowers the price' with little or no loss of accuracy. Encode
business logic as constraints whenever you can.""")

step("3.4  Feature importance: impurity-based vs permutation")
noisy = tr.assign(listing_id_hash=np.random.default_rng(5).random(len(tr)))      # unique random number per row
noisy_te = te.assign(listing_id_hash=np.random.default_rng(6).random(len(te)))
F4 = HOUSE_FEATURES + ["listing_id_hash"]
deep = DecisionTreeRegressor(random_state=0).fit(noisy[F4], noisy.price)               # fully grown
perm = permutation_importance(deep, noisy_te[F4], noisy_te.price, n_repeats=10, random_state=0)
imp = pd.DataFrame({"impurity importance %": 100 * deep.feature_importances_,
                    "permutation importance (test R^2 drop)": perm.importances_mean}, index=F4)
imp["impurity rank"] = imp["impurity importance %"].rank(ascending=False).astype(int)
imp["permutation rank"] = imp["permutation importance (test R^2 drop)"].rank(ascending=False).astype(int)
print(imp.sort_values("impurity importance %", ascending=False).round(3).to_string())
note(f"""Impurity importance (feature_importances_) measures how much each feature was
USED to reduce TRAINING error. A pure random number ranks #{imp.loc['listing_id_hash', 'impurity rank']}, ahead of three
real price drivers, because a fully grown tree can always use a unique value to
split off noisy rows. Permutation importance on held-out data asks the right
question: 'how much worse are test predictions if this column is shuffled?'
The random column scores ~0. Never present impurity importance to stakeholders
as 'what drives the price'.""")

banner("KEY TAKEAWAYS")
print("""  1. A regression tree predicts the mean of a leaf: a step function, built by greedy MSE splits.
  2. No scaling needed; thresholds and interactions are found automatically.
  3. Control it with min_samples_leaf / max_depth or cost-complexity pruning (ccp_alpha).
  4. Trees cannot extrapolate: never use a single tree on trending targets without handling the trend.
  5. Single trees are unstable; use monotonic constraints and permutation importance in production.""")
