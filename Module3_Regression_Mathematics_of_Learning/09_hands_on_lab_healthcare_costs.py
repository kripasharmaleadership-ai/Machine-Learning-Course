"""
LAB 09 - HANDS-ON LAB: Predict Yearly Healthcare Costs
======================================================

THE BUSINESS PROBLEM
    A health insurer wants to estimate each member's yearly medical charges from the
    information available at enrolment: age, sex, BMI, number of children, smoking
    status and region. Uses: premium setting, budgeting, and care-management outreach.
    The actuarial team must be able to EXPLAIN every price driver to the regulator.

THE DATA (real): Medical Cost Personal dataset, 1,338 US insurance members
    (downloaded once from OpenML, cached in data/medical_costs.csv)

WHAT YOU WILL DO
    Part 1 (EASY)   Explore the data and the target
    Part 2 (EASY)   Split once, then build BASELINES
    Part 3 (MEDIUM) Train LINEAR REGRESSION inside a pipeline and read its coefficients
    Part 4 (MEDIUM) Apply REGULARISATION (Ridge, Lasso, Elastic Net), also on 44 interaction features
    Part 5 (HARD)   COMPARE models with repeated CV, ANALYSE RESIDUALS, fix what they reveal,
                    INTERPRET the important features, open the test set once, save the model
"""
# %%
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml_lab_utils import (HAS_MPL, MEDICAL_TARGET, OUT_DIR, RANDOM_STATE, banner, load_medical_costs, note, plt,
                          save_fig, show, step)
from regression_toolkit import bootstrap_compare, regression_metrics, residual_table

TARGET = MEDICAL_TARGET
NUM = ["age", "bmi", "children"]
CAT = ["sex", "smoker", "region"]

# %% =========================================================================
# PART 1 (EASY): Explore
# ============================================================================
banner("PART 1 - Explore the data", "EASY")
raw = load_medical_costs()
show("Rows x columns", raw.shape)
print(raw.head().to_string())
print(raw.describe(include="all").T[["count", "unique", "mean", "std", "min", "max"]].to_string())
show("Missing values (total)", int(raw.isna().sum().sum()))
dups = raw[raw.duplicated(keep=False)]
show("Exact duplicate rows", int(raw.duplicated().sum()))
print(dups.to_string())
df = raw.drop_duplicates().reset_index(drop=True)
note("""One exact duplicate: two members with identical age, sex, BMI to 3 decimals,
children, region AND charges to the cent. Almost certainly a double-loaded
record. We drop it, so it cannot sit in train and test at the same time
(Module 2, Lab 02).""")

step("1.1  The target: yearly charges")
show("Mean / median charges", f"${df[TARGET].mean():,.0f} / ${df[TARGET].median():,.0f}")
show("Skewness (0 = symmetric)", float(skew(df[TARGET])))
print(df.groupby("smoker")[TARGET].agg(["count", "mean", "median"]).round(0).to_string())
print(df[NUM + [TARGET]].corr()[TARGET].round(3).drop(TARGET).rename("correlation with charges").to_string())
note("""Right-skewed money target (mean well above median). Smokers cost ~4x more:
smoking is by far the strongest single signal. Age correlates moderately, BMI
weakly, children barely. Correlation only sees straight lines (Lab 01, 2.4),
so the plots matter.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df[TARGET] / 1000, bins=50, color="#4c78a8")
    axes[0].set(xlabel="charges ($k)", title=f"Target: skew {skew(df[TARGET]):.2f}")
    for s, c in [("no", "#4c78a8"), ("yes", "#e45756")]:
        g = df[df.smoker == s]
        axes[1].scatter(g.age, g[TARGET] / 1000, s=6, color=c, alpha=0.6, label=f"smoker={s}")
        axes[2].scatter(g.bmi, g[TARGET] / 1000, s=6, color=c, alpha=0.6, label=f"smoker={s}")
    axes[1].set(xlabel="age", ylabel="charges ($k)", title="Three bands of cost by age")
    axes[2].axvline(30, color="k", ls=":")
    axes[2].set(xlabel="BMI", title="Smokers jump at BMI 30")
    axes[1].legend()
    save_fig(fig, "09_part1_eda")
note("""Look at the age plot: THREE parallel bands, not one. And in the BMI plot,
smokers' costs jump at BMI = 30 (the clinical obesity threshold) while
non-smokers' don't. Keep these two pictures in mind: the residuals in Part 5
will rediscover them.""")

# %% =========================================================================
# PART 2 (EASY): Split once, then baselines
# ============================================================================
banner("PART 2 - Split, then create baselines", "EASY")
strata = pd.qcut(df[TARGET], 5, labels=False)                      # stratify a continuous target by quintile
train, test = train_test_split(df, test_size=0.2, stratify=strata, random_state=RANDOM_STATE)
X_tr, y_tr, X_te, y_te = train[NUM + CAT], train[TARGET], test[NUM + CAT], test[TARGET]
show("Train / test rows", f"{len(train)} / {len(test)}")
show("Mean charges train / test", f"${y_tr.mean():,.0f} / ${y_te.mean():,.0f}")
note("""We stratify on target quintiles so both sets contain the same mix of cheap and
very expensive members. The TEST SET IS NOW SEALED: everything up to Part 5.5
uses only training data and cross-validation.""")

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)


class GroupMean(BaseEstimator):
    """Baseline 'business rule': predict the average charges of the member's smoker group."""

    def fit(self, X, y):
        self.means_ = pd.Series(np.asarray(y)).groupby(np.asarray(X["smoker"])).mean()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, X):
        return X["smoker"].map(self.means_).fillna(self.global_).to_numpy()


def cv_scores(model, X=X_tr, y=y_tr):
    s = cross_validate(model, X, y, cv=cv, scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"))
    return {"CV RMSE": -s["test_neg_root_mean_squared_error"].mean(), "CV MAE": -s["test_neg_mean_absolute_error"].mean(),
            "CV R2": s["test_r2"].mean(), "RMSE sd": s["test_neg_root_mean_squared_error"].std()}


results = {
    "B0 predict the mean": cv_scores(DummyRegressor(strategy="mean")),
    "B1 predict the median": cv_scores(DummyRegressor(strategy="median")),
    "B2 mean per smoker group": cv_scores(GroupMean()),
}
print(pd.DataFrame(results).T.round(3).to_string())
note("""Any model must beat these to be worth its complexity. B0 has R^2 = 0 by
definition. The median has a LOWER MAE but a higher RMSE than the mean
(Lab 02: MAE is minimised by the median). The one-line rule 'smoker or not'
already explains ~60% of the variance: the bar for machine learning is high.""")

# %% =========================================================================
# PART 3 (MEDIUM): Linear regression in a pipeline
# ============================================================================
banner("PART 3 - Train linear regression", "MEDIUM")


def preprocessor(scale=True):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT),
    ], verbose_feature_names_out=False)


linear = Pipeline([("prep", preprocessor(scale=False)), ("model", LinearRegression())])
results["M1 linear regression"] = cv_scores(linear)
print(pd.DataFrame(results).T.round(3).to_string())

linear.fit(X_tr, y_tr)
coef = pd.Series(linear["model"].coef_, index=linear["prep"].get_feature_names_out())
print("\n  Coefficients ($ per unit, holding the others fixed):")
print(coef.round(0).to_string())
show("Intercept", f"${linear['model'].intercept_:,.0f}")
note("""Without scaling, coefficients read in business units: each year of age adds
~$260 a year, each BMI point ~$330, being a smoker ~$24,000 (vs a non-smoker of
the same age, BMI...). Categorical coefficients are relative to the dropped
category (female, northeast). Sex and region are small next to smoking. This is
already an explainable pricing model, and it beats every baseline.""")

# %% =========================================================================
# PART 4 (MEDIUM): Regularisation
# ============================================================================
banner("PART 4 - Apply regularisation", "MEDIUM")
alphas = np.logspace(-3, 4, 50)
reg_models = {
    "M2 RidgeCV": RidgeCV(alphas=alphas),
    "M3 LassoCV": LassoCV(alphas=alphas, cv=KFold(5, shuffle=True, random_state=0), max_iter=50000),
    "M4 ElasticNetCV": ElasticNetCV(l1_ratio=[.1, .5, .9], alphas=alphas, cv=KFold(5, shuffle=True, random_state=0),
                                    max_iter=50000),
}
for name, m in reg_models.items():
    results[name] = cv_scores(Pipeline([("prep", preprocessor()), ("model", m)]))
print(pd.DataFrame(results).T.round(3).to_string())
note("""Honest result: regularisation changes almost nothing here. 8 features and
~1,070 rows means low variance already (Lab 04): there is nothing for the
penalty to fix. Regularisation pays off when the model is FLEXIBLE relative to
the data. So let's make it flexible.""")

step("4.1  All squares and pairwise interactions (44 features): where regularisation earns its keep")


def poly_pipe(model):
    return Pipeline([("prep", preprocessor()), ("poly", PolynomialFeatures(2, include_bias=False)),
                     ("scale", StandardScaler()), ("model", model)])


results["M5 OLS + all interactions"] = cv_scores(poly_pipe(LinearRegression()))
results["M6 Lasso + all interactions"] = cv_scores(poly_pipe(LassoCV(cv=KFold(5, shuffle=True, random_state=0),
                                                                     max_iter=100000)))
print(pd.DataFrame(results).T.round(3).to_string())
lasso_poly = poly_pipe(LassoCV(cv=KFold(5, shuffle=True, random_state=0), max_iter=100000)).fit(X_tr, y_tr)
names = lasso_poly["poly"].get_feature_names_out(lasso_poly["prep"].get_feature_names_out())
lc = pd.Series(lasso_poly["model"].coef_, index=names)
print("\n  Lasso: largest coefficients among 44 candidate features ($ per sd):")
print(lc[lc.abs() > 1e-6].sort_values(key=abs, ascending=False).head(6).round(0).to_string())
show("Features kept by Lasso", f"{(lc.abs() > 1e-6).sum()} of {len(lc)}")
note("""Squares and pairwise products of every feature: the model can now bend and
interact. The biggest jump in accuracy of the lab so far, and after smoking
itself, Lasso's strongest feature is the interaction 'bmi x smoker_yes': it
found the pattern of the BMI plot by itself, and switched ~half of the 44
candidates off. Its weakness: a product bmi x smoker is a straight line in BMI, but the
plot showed a JUMP at 30. Residuals will tell us what is still missing.""")

# %% =========================================================================
# PART 5 (HARD): Compare, analyse residuals, interpret
# ============================================================================
banner("PART 5 - Compare models, analyse residuals, interpret", "HARD")

step("5.1  A decision tree for comparison")
tree = GridSearchCV(DecisionTreeRegressor(random_state=0),
                    {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [5, 10, 20, 40]},
                    cv=KFold(5, shuffle=True, random_state=0), scoring="neg_root_mean_squared_error")
tree_pipe = Pipeline([("prep", preprocessor(scale=False)), ("model", tree)])
tree_pipe.fit(X_tr, y_tr)
best_tree = Pipeline([("prep", preprocessor(scale=False)),
                      ("model", DecisionTreeRegressor(random_state=0, **tree.best_params_))])
results[f"M7 decision tree {tree.best_params_}"] = cv_scores(best_tree)
print(pd.DataFrame(results).T.round(3).to_string())

step("5.2  Residual analysis of the plain linear model (out-of-fold predictions)")
oof = np.zeros(len(X_tr))
for tr_i, va_i in KFold(5, shuffle=True, random_state=1).split(X_tr):
    oof[va_i] = clone(linear).fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i]).predict(X_tr.iloc[va_i])
res = y_tr.to_numpy() - oof
segment = (np.where(train.smoker == "yes", "smoker", "non-smoker") + " / "
           + np.where(train.bmi >= 30, "BMI>=30", "BMI<30"))
print(residual_table(y_tr, oof, pd.Series(segment)).round(0).to_string())
note("""Residuals = what the model got wrong, measured on data it did not train on.
They are NOT random: the linear model over-charges smokers with BMI < 30 and
under-charges obese smokers by a similar amount (mean residuals of opposite
sign). It fitted ONE smoking premium, but the data has TWO: one for smokers and a
much larger one for obese smokers. A pattern in the residuals is a feature you
have not built yet.""")

step("5.3  Fix: two domain features suggested by the residuals")


class HealthFeatures(TransformerMixin, BaseEstimator):
    """Domain features: obese smoker (BMI >= 30 AND smoker), and age^2 (costs accelerate with age)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["smoker_obese"] = ((X["smoker"] == "yes") & (X["bmi"] >= 30)).astype(float)
        X["age_sq"] = X["age"].astype(float) ** 2
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(list(input_features) + ["smoker_obese", "age_sq"])


NUM_FE = NUM + ["age_sq", "smoker_obese"]


def fe_pipe(model, scale=True):
    prep = ColumnTransformer([("num", StandardScaler() if scale else "passthrough", NUM_FE),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT)],
                             verbose_feature_names_out=False)
    return Pipeline([("features", HealthFeatures()), ("prep", prep), ("model", model)])


results["M8 linear + domain features"] = cv_scores(fe_pipe(LinearRegression(), scale=False))
results["M9 Ridge + domain features"] = cv_scores(fe_pipe(RidgeCV(alphas=alphas)))
comparison = pd.DataFrame(results).T.sort_values("CV RMSE")
print(comparison.round(3).to_string())
note("""Two features, chosen by looking at the residuals and grounded in medicine
(the obesity threshold, costs accelerating with age), beat 44 automatic
interactions AND the tuned tree. They also keep the model a short, explainable
formula. This is the central lesson of regression in practice: the residual
plot tells you what to engineer next.""")

oof_fe = np.zeros(len(X_tr))
final_template = fe_pipe(LinearRegression(), scale=False)
for tr_i, va_i in KFold(5, shuffle=True, random_state=1).split(X_tr):
    oof_fe[va_i] = clone(final_template).fit(X_tr.iloc[tr_i], y_tr.iloc[tr_i]).predict(X_tr.iloc[va_i])
print(residual_table(y_tr, oof_fe, pd.Series(segment)).round(0).to_string())
note("""After the fix, mean residuals are close to zero in every segment: no more
systematic over- or under-charging of any group. What remains is a smaller
group of non-smokers with charges far above prediction (long right tail of
residuals): costs caused by things we have no column for (chronic conditions,
accidents). More modelling can't fix missing information.""")

step("5.4  Interpret the important features")
final = fe_pipe(LinearRegression(), scale=False).fit(X_tr, y_tr)
coef = pd.Series(final["model"].coef_, index=final["prep"].get_feature_names_out())
perm = permutation_importance(final, X_tr, y_tr, n_repeats=10, random_state=0, scoring="neg_root_mean_squared_error")
interp = pd.DataFrame({"coefficient ($)": coef.round(1)})
perm_s = pd.Series(perm.importances_mean, index=X_tr.columns).sort_values(ascending=False)
print(interp.to_string())
print("\n  Permutation importance on raw inputs (RMSE increase in $ when shuffled):")
print(perm_s.round(0).to_string())
age40 = coef["age"] * 40 + coef["age_sq"] * 1600
age60 = coef["age"] * 60 + coef["age_sq"] * 3600
note(f"""In plain words for the actuarial team:
- A smoker with BMI < 30 costs ~${coef['smoker_yes']:,.0f} a year more than a non-smoker
  with the same profile; a smoker with BMI >= 30 costs an EXTRA ~${coef['smoker_obese']:,.0f} on top.
- Age: going from 40 to 60 adds ~${age60 - age40:,.0f} a year, and the effect accelerates.
- BMI itself (outside the smoker threshold) and children matter far less;
  sex and region are small, and might be dropped for regulatory reasons
  (many jurisdictions forbid pricing on sex) at very little accuracy cost.
Permutation importance (shuffle one raw input, measure the damage) ranks
smoker >> bmi > age >> the rest. BMI ranks high although its own coefficient
is tiny: it matters THROUGH the obese-smoker threshold. Coefficients and
permutation importance answer different questions; report both.""", keep_lines=True)

step("5.5  Open the test set ONCE")
champion = fe_pipe(LinearRegression(), scale=False).fit(X_tr, y_tr)
challengers = {"B2 mean per smoker group": GroupMean().fit(X_tr, y_tr), "M1 linear regression": linear.fit(X_tr, y_tr),
               "M7 decision tree": best_tree.fit(X_tr, y_tr), "M8 linear + domain features (champion)": champion}
n_inputs = {"B2 mean per smoker group": 1, "M1 linear regression": 8, "M7 decision tree": 8,
            "M8 linear + domain features (champion)": len(champion["prep"].get_feature_names_out())}
test_tab = pd.DataFrame({k: regression_metrics(y_te, m.predict(X_te), n_features=n_inputs[k])
                         for k, m in challengers.items()}).T
print(test_tab.round(3).to_string(float_format=lambda v: f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"))
ci = bootstrap_compare(y_te, champion.predict(X_te), best_tree.predict(X_te))
show("RMSE(tree) - RMSE(champion), 95% CI", f"${ci['diff (B - A)']:,.0f}  [{ci['ci_low']:,.0f}, {ci['ci_high']:,.0f}]")
show("Champion test R^2 / adjusted R^2", f"{test_tab.iloc[-1]['R2']:.3f} / {test_tab.iloc[-1]['Adj R2']:.3f}")
note("""The test results confirm the CV ranking. The champion's lead over the tree is
small; check the bootstrap interval before claiming it is real. If the
interval includes 0, the tie-breaker is the business requirement: the linear
model is a formula the regulator can audit, the tree is a set of steps.""")

step("5.6  Save the model, its metrics and a model card")
champion_full = fe_pipe(LinearRegression(), scale=False).fit(X_tr, y_tr)
joblib.dump(champion_full, OUT_DIR / "medical_cost_model_v1.joblib")
metrics = {"model": "linear regression + domain features (smoker_obese, age_sq)",
           "train_rows": len(train), "test_rows": len(test), "cv": comparison.loc["M8 linear + domain features"].to_dict(),
           "test": test_tab.iloc[-1].to_dict(), "coefficients": coef.round(2).to_dict(),
           "intercept": float(champion_full["model"].intercept_)}
(OUT_DIR / "medical_cost_metrics_v1.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
card = f"""# Model Card: Yearly Medical Charges v1

- **Use:** estimate yearly charges per member at enrolment (pricing, budgeting, outreach)
- **Data:** Medical Cost Personal dataset (1,338 US members; 1 duplicate removed); 80/20 split stratified on target quintiles
- **Model:** linear regression on age, age^2, BMI, children, sex, smoker, region, smoker x (BMI >= 30)
- **Test performance:** RMSE ${test_tab.iloc[-1]['RMSE']:,.0f}, MAE ${test_tab.iloc[-1]['MAE']:,.0f}, R^2 {test_tab.iloc[-1]['R2']:.3f}
- **Baseline (mean per smoker group):** RMSE ${test_tab.iloc[0]['RMSE']:,.0f}, R^2 {test_tab.iloc[0]['R2']:.3f}
- **Main drivers:** smoking (+${coef['smoker_yes']:,.0f}), obese smoker (+${coef['smoker_obese']:,.0f} more), age (accelerating)
- **Known limitations:** no medical history columns -> large unexplained costs for some non-smokers;
  1,338 members from one source, may not represent other populations; linear in age^2 beyond 64 is untested
- **Fairness note:** `sex` has a small coefficient; check local regulation before using it in pricing
- **Do not use for:** individual clinical decisions; members outside age 18-64
"""
(OUT_DIR / "medical_cost_model_card.md").write_text(card, encoding="utf-8")
print("  [saved] outputs/medical_cost_model_v1.joblib, medical_cost_metrics_v1.json, medical_cost_model_card.md")

if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    seg_colors = {"non-smoker / BMI<30": "#4c78a8", "non-smoker / BMI>=30": "#72b7b2",
                  "smoker / BMI<30": "#f58518", "smoker / BMI>=30": "#e45756"}
    for s, c in seg_colors.items():
        mk = segment == s
        axes[0, 0].scatter(oof[mk] / 1000, res[mk] / 1000, s=6, color=c, label=s)
        axes[0, 1].scatter(oof_fe[mk] / 1000, (y_tr.to_numpy() - oof_fe)[mk] / 1000, s=6, color=c, label=s)
    for ax, t in [(axes[0, 0], "Residuals: plain linear model"), (axes[0, 1], "Residuals: + 2 domain features")]:
        ax.axhline(0, color="k")
        ax.set(xlabel="predicted charges ($k)", ylabel="residual ($k)", title=t, ylim=(-25, 40))
    axes[0, 0].legend(fontsize=7)
    comp = comparison["CV RMSE"].sort_values()
    axes[1, 0].barh(comp.index, comp.values, color=["#54a24b" if "domain" in i else "#9d9d9d" for i in comp.index])
    axes[1, 0].set(xlabel="CV RMSE ($)", title="Model comparison (repeated 5-fold CV)")
    axes[1, 0].invert_yaxis()
    p_te = champion.predict(X_te)
    axes[1, 1].scatter(p_te / 1000, y_te / 1000, s=8, alpha=0.6)
    lim = [0, max(p_te.max(), y_te.max()) / 1000]
    axes[1, 1].plot(lim, lim, "k--")
    axes[1, 1].set(xlabel="predicted ($k)", ylabel="actual ($k)", title="Champion on the test set")
    save_fig(fig, "09_part5_residuals_and_comparison")

banner("KEY TAKEAWAYS")
print("""  1. Baselines first: 'smoker or not' explains ~60% of the variance on its own.
  2. Linear regression in a pipeline gives an explainable, $-denominated pricing formula.
  3. Regularisation adds little to a small model; it matters once the model gets flexible.
  4. Residuals revealed the missing pattern: two domain features beat 44 interactions and a tree.
  5. Open the test set once, compare with a CI, and ship the model with metrics and a model card.""")
