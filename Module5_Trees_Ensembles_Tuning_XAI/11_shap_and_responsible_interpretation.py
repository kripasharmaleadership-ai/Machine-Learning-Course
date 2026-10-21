"""
LAB 11 - SHAP and Responsible Model Interpretation
==================================================

SHAPLEY VALUES (Shapley, 1953 -> Lundberg & Lee's SHAP, 2017)
    A fair way to split a payout among players who cooperate. Here: the "payout" is
        f(x) - E[f(X)]   (this prediction minus the average prediction),
    the "players" are the features. A feature's Shapley value = its average marginal
    contribution over every order in which features could be revealed:
        phi_i = sum over coalitions S without i of  |S|! (p-|S|-1)! / p!  x  [v(S + i) - v(S)]
        v(S) = average prediction when only the features in S are known (others from a background set)

    Guarantees: EFFICIENCY (the phis add up exactly to f(x) - E[f(X)]), SYMMETRY (identical
    contributions get identical credit), DUMMY (an unused feature gets 0), ADDITIVITY.

THE `shap` LIBRARY approximates or speeds this up (TreeSHAP, KernelSHAP). Here we compute it EXACTLY
by enumerating all coalitions (tree_toolkit.shapley_exact), which is feasible for <= ~11 features.

RESPONSIBLE INTERPRETATION (the HARD level)
    SHAP explains the MODEL, relative to a chosen background population. It does not say what
    CAUSES the outcome, and it changes when features are correlated, when another equally good
    model is chosen, or when the background changes.

LEVELS
    EASY   : Shapley values by hand for a 3-feature model with an interaction; the four guarantees.
    MEDIUM : Exact SHAP for the bike model; checked against the TRUE formula; local and global views.
    HARD   : Correlated twins split credit; the Rashomon effect (equally good models, different
             reasons); the background population changes the explanation.
"""
# %%
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (BIKE_FEATURES, CREDIT_NUM, HAS_MPL, RANDOM_STATE, banner, bike_components, make_bike_rentals,
                          make_credit_applicants, note, optional_import, plt, save_fig, show, step)
from tree_toolkit import shapley_exact

# %% =========================================================================
# LEVEL 1 (EASY): Shapley values by hand
# ============================================================================
banner("Three features, eight coalitions, six orderings", "EASY")


def toy(df):
    """A tiny risk score (in percentage points) with an interaction: high debt AND late payments."""
    hd, lp, yg = (df.debt_ratio > 0.45).astype(float), (df.late_payments > 0).astype(float), (df.age < 30).astype(float)
    return (5 + 20 * hd + 10 * lp + 5 * yg + 30 * hd * lp).to_numpy()


background = pd.DataFrame({"debt_ratio": [0.2, 0.3, 0.6, 0.25], "late_payments": [0, 0, 1, 0], "age": [45, 30, 52, 38]})
x = pd.DataFrame({"debt_ratio": [0.7], "late_payments": [2], "age": [24]})
feats = list(x.columns)
print("  Background (the 'average' applicants):")
print(background.to_string(index=False))
print(f"\n  Applicant to explain: {x.iloc[0].to_dict()}   score f(x) = {toy(x)[0]:.0f}")


def v(S):
    """Average score when the features in S are set to the applicant's values, the rest from the background."""
    return toy(background.assign(**{f: x[f].iloc[0] for f in S})).mean()


rows = {}
for k in range(4):
    from itertools import combinations
    for S in combinations(feats, k):
        rows["{" + ", ".join(S) + "}" if S else "{} (nothing known)"] = v(S)
print("\n  v(S) for every coalition:")
print(pd.Series(rows).round(2).to_string())
contrib = {f: [] for f in feats}
for order in permutations(feats):
    known = []
    for f in order:
        contrib[f].append(v(known + [f]) - v(known))
        known.append(f)
phi = pd.Series({f: np.mean(c) for f, c in contrib.items()})
print("\n  Average marginal contribution over the 6 orderings = Shapley value:")
print(pd.DataFrame({"contributions in the 6 orderings": {f: np.round(c, 1).tolist() for f, c in contrib.items()},
                    "Shapley value": phi.round(2)}).to_string())
show("base value E[f] + sum of Shapley values", f"{v([]):.2f} + {phi.sum():.2f} = {v([]) + phi.sum():.2f}  (f(x) = {toy(x)[0]:.2f})")
show("Same values with tree_toolkit.shapley_exact", shapley_exact(toy, x, background).round(2).to_dict())
note("""Reveal the features one at a time in every possible order; record how much
each one moves the average score when it arrives; average over the orders.
The interaction (high debt AND late payments: +30) is shared between debt and
late payments, because neither creates it alone. The values add up EXACTLY to
'this applicant minus the average applicant': no unexplained remainder.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Exact SHAP for a real model, checked against the truth
# ============================================================================
banner("SHAP for the bike model, against the true formula", "MEDIUM")
bikes = make_bike_rentals(n=10_000, seed=10, with_truth=True)
X, y = bikes[BIKE_FEATURES], bikes["rentals"].to_numpy()
X_tr, X_te, y_tr, y_te, b_tr, b_te = train_test_split(X, y, bikes, test_size=0.2, random_state=RANDOM_STATE)
NO_TWIN = [f for f in BIKE_FEATURES if f != "feels_like_c"]
model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(X_tr[NO_TWIN], y_tr)
bg = X_tr[NO_TWIN].sample(100, random_state=0)
GROUPS = {"hour & workday": ["hour", "workday"], "temp_c": ["temp_c"], "humidity": ["humidity"], "wind_kmh": ["wind_kmh"],
          "rain": ["rain"]}
true_bg = bike_components(bikes.loc[bg.index]).mean()


def true_contrib(row_idx):
    """The truth's own 'SHAP': each true component minus its average over the background."""
    c = bike_components(bikes.loc[[row_idx]]).iloc[0] - true_bg
    return pd.Series({"hour & workday": c["hour_x_workday"], "temp_c": c["temp_c"], "humidity": c["humidity"],
                      "wind_kmh": 0.0, "rain": c["rain"]})


step("2.1  One prediction, explained (a local explanation)")
row = X_te[NO_TWIN].iloc[[11]]
phi_row = shapley_exact(model.predict, row, bg, GROUPS)
print(row.to_string())
print(pd.DataFrame({"SHAP (model)": phi_row, "TRUE contribution": true_contrib(row.index[0])}).round(1).to_string())
show("base value + sum(SHAP) = prediction", f"{phi_row.attrs['base_value']:.1f} + {phi_row.sum():.1f} = "
                                           f"{phi_row.attrs['base_value'] + phi_row.sum():.1f}  (model: {model.predict(row)[0]:.1f})")
print("\n  Waterfall, as you would show it to a planner:")
running = phi_row.attrs["base_value"]
print(f"    average hour ............................ {running:7.0f} rentals")
for f_, val in phi_row.sort_values(key=abs, ascending=False).items():
    running += val
    print(f"    {f_ + ' ' + ('raises' if val > 0 else 'lowers'):<38} {val:+7.0f}  -> {running:.0f}")
note("""Because the data follows a known additive formula, each feature's TRUE
contribution is known too: its component for this hour minus the component's
average. The model's SHAP values land close to them: SHAP recovered WHY this
hour is busy or quiet, feature by feature, and the waterfall adds up exactly
to the prediction.""")

step("2.2  Many local explanations = a global view")
sample = X_te[NO_TWIN].sample(60, random_state=1)
shap_tab = pd.DataFrame([shapley_exact(model.predict, sample.iloc[[i]], bg, GROUPS) for i in range(len(sample))],
                        index=sample.index)
truth_tab = pd.DataFrame([true_contrib(i) for i in sample.index], index=sample.index)
glob = pd.DataFrame({"mean |SHAP| (model)": shap_tab.abs().mean(), "mean |TRUE contribution|": truth_tab.abs().mean(),
                     "correlation SHAP vs truth": [np.corrcoef(shap_tab[c], truth_tab[c])[0, 1] if truth_tab[c].std() > 0
                                                   else np.nan for c in shap_tab.columns]})
print(glob.round(3).to_string())
shap_lib = optional_import("shap")
if shap_lib is not None:
    ex = shap_lib.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    print("  shap.TreeExplainer (installed) on the same row:", np.round(ex.shap_values(row)[0], 1).tolist())
note("""Averaging |SHAP| over many rows gives a global importance ranking that is
grounded in local explanations, and here it matches the true formula's
ranking: hour & workday >> temperature > rain, humidity, and wind ~0. The
per-feature correlation with the TRUE contributions is 0.99 or better for the
real drivers. (If the `shap` package is installed, its TreeExplainer gives
the same numbers faster.)""")

# %% =========================================================================
# LEVEL 3 (HARD): Responsible interpretation
# ============================================================================
banner("What SHAP can and cannot tell you", "HARD")

step("3.1  Correlated twins split the credit")
twin_model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(X_tr, y_tr)
bg2 = X_tr.sample(100, random_state=0)
hot = X_te[X_te.temp_c.between(23, 25)].iloc[[0]]
split = shapley_exact(twin_model.predict, hot, bg2, {"hour & workday": ["hour", "workday"], "temp_c": ["temp_c"],
                                                     "feels_like_c": ["feels_like_c"], "humidity": ["humidity"],
                                                     "wind_kmh": ["wind_kmh"], "rain": ["rain"]})
grouped = shapley_exact(twin_model.predict, hot, bg2, {"hour & workday": ["hour", "workday"],
                                                       "temperature (both columns)": ["temp_c", "feels_like_c"],
                                                       "humidity": ["humidity"], "wind_kmh": ["wind_kmh"], "rain": ["rain"]})
show("A 24 C hour: SHAP temp_c / feels_like_c", f"{split['temp_c']:+.1f} / {split['feels_like_c']:+.1f}")
show("Same hour, the two columns grouped", f"{grouped['temperature (both columns)']:+.1f}")
show("TRUE temperature contribution", f"{(bike_components(bikes.loc[hot.index]).iloc[0]['temp_c'] - bike_components(bikes.loc[bg2.index])['temp_c'].mean()):+.1f}")
note("""With two near-identical temperature columns the model uses both, and SHAP
splits the credit: feels_like_c gets a sizeable share although it has no
effect at all. Each number alone understates temperature. Grouped, the two
columns get the full (correct) credit. Before explaining, find correlated
features and explain them as a GROUP.""")

step("3.2  The Rashomon effect: equally good models, different reasons")
credit = make_credit_applicants(n=12_000, seed=11, with_truth=True)
Xc = credit[CREDIT_NUM].assign(employment_risk=credit.employment.map({"salaried": 0, "self_employed": 1, "contract": 2,
                                                                       "unemployed": 3}))
yc = credit["default"].to_numpy()
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=RANDOM_STATE)
no_util = [c for c in Xc.columns if c != "utilisation"]
models = {
    "gradient boosting": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0), list(Xc.columns)),
    "random forest": (RandomForestClassifier(300, min_samples_leaf=20, n_jobs=-1, random_state=0), list(Xc.columns)),
    "logistic regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), list(Xc.columns)),
    "boosting, utilisation not collected": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0), no_util),
}
applicant = Xc_te[(Xc_te.age < 30) & (Xc_te.utilisation > 0.8)].iloc[[0]]
bg_c = Xc_tr.sample(80, random_state=0)
reasons, aucs = {}, {}
for name, (m, cols) in models.items():
    m.fit(Xc_tr[cols], yc_tr)
    aucs[name] = roc_auc_score(yc_te, m.predict_proba(Xc_te[cols])[:, 1])
    phi_c = shapley_exact(lambda d: m.predict_proba(d)[:, 1], applicant[cols], bg_c[cols])
    top3 = phi_c.sort_values(ascending=False).head(3)
    reasons[name] = {"test AUC": round(aucs[name], 3), "P(default) for this applicant": round(phi_c.attrs["prediction"], 3),
                     "top 3 reasons (SHAP, in probability)": ", ".join(f"{k} {v_:+.3f}" for k, v_ in top3.items())}
print("  Applicant:", applicant.iloc[0].round(3).to_dict())
print(pd.DataFrame(reasons).T.to_string())
probs = [r_["P(default) for this applicant"] for r_ in reasons.values()]
note(f"""Plot twist: four models with test AUC within {max(aucs.values()) - min(aucs.values()):.2f} of each other give this
applicant a default probability anywhere from {min(probs):.2f} to {max(probs):.2f}, and DIFFERENT top reasons.
In credit, those reasons go into the legally required 'adverse action'
notice, so the explanation a customer receives depends on a modelling choice
that accuracy could not distinguish (the Rashomon set). Without utilisation,
open_cards (no real effect) appears among the reasons: a proxy explanation.
Responsible practice: check that explanations are stable across the
near-equivalent models before you rely on them, and prefer models whose
reasons are causally sensible.""")

step("3.3  'Compared with whom?' The background population changes the answer")
gb, cols = models["gradient boosting"]
backgrounds = {"all applicants": Xc_tr.sample(80, random_state=1),
               "approved, low-risk applicants": Xc_tr[gb.predict_proba(Xc_tr[cols])[:, 1] < 0.08].sample(80, random_state=1),
               "applicants of the same age band (< 30)": Xc_tr[Xc_tr.age < 30].sample(80, random_state=1)}
bg_tab = {}
for name, b in backgrounds.items():
    ph = shapley_exact(lambda d: gb.predict_proba(d)[:, 1], applicant[cols], b[cols])
    bg_tab[name] = {"base value E[f]": ph.attrs["base_value"], **ph.round(3).to_dict()}
bg_df = pd.DataFrame(bg_tab).T
print(bg_df.round(3).to_string())
note(f"""SHAP explains 'this applicant versus a REFERENCE group'. Against all
applicants, against approved low-risk ones, or against people of the same age,
the base value and the contributions change: 'age' even drops to about zero
when the comparison group is the same age, and the applicant's moderate debt
counts in their FAVOUR ({bg_df.iloc[0]['debt_ratio']:+.2f}) against all applicants but as nothing ({bg_df.iloc[1]['debt_ratio']:+.2f}) against
low-risk ones. There is no neutral background. State it
in every explanation ('compared with a typical approved applicant, your
utilisation raised the estimated risk by...'), and choose it to match the
question the reader is asking.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for c, col in zip(shap_tab.columns, ["#4c78a8", "#e45756", "#54a24b", "#9d9d9d", "#f58518"]):
        axes[0].scatter(truth_tab[c], shap_tab[c], s=12, color=col, label=c)
    lim = [min(shap_tab.min().min(), truth_tab.min().min()), max(shap_tab.max().max(), truth_tab.max().max())]
    axes[0].plot(lim, lim, "k:", lw=1)
    axes[0].set(xlabel="TRUE contribution (from the formula)", ylabel="SHAP value (from the model)",
                title="SHAP recovers the true contributions")
    axes[0].legend(fontsize=8)
    w = phi_row.sort_values()
    axes[1].barh(w.index, w.values, color=["#e45756" if v_ < 0 else "#54a24b" for v_ in w.values])
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set(xlabel=f"contribution (rentals), base value {phi_row.attrs['base_value']:.0f}",
                title=f"One hour explained: prediction {model.predict(row)[0]:.0f}")
    save_fig(fig, "11_shap")

banner("KEY TAKEAWAYS")
print("""  1. A Shapley value = a feature's average marginal contribution over all orderings of the features.
  2. SHAP values add up exactly to (prediction - average prediction): a complete local accounting.
  3. Averaging |SHAP| over many rows gives a global importance grounded in local explanations.
  4. Correlated features split credit (group them); equally good models can give different reasons.
  5. Explanations are relative to a background group and describe the model, not causes: say so.""")
