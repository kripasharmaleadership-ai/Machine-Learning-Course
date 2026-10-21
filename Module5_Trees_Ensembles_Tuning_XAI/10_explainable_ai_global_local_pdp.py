"""
LAB 10 - Explainable AI: Global vs Local Explanations, Partial Dependence and ICE
================================================================================

TWO QUESTIONS PEOPLE ASK ABOUT A MODEL
    GLOBAL : "How does the model behave overall? What drives its predictions?"
             -> feature importance (Lab 08), partial dependence (PDP), accumulated local effects (ALE)
    LOCAL  : "Why did the model predict THIS for THIS case?"
             -> the tree path (Lab 02), ICE curves, what-if analysis, SHAP (Lab 11)

PARTIAL DEPENDENCE (Friedman, 2001)
    PDP_j(v) = average over all rows of f(x with feature j set to v)
    "What does the model predict on average if everybody had temperature v?"
    ICE = the same curve for each row separately (PDP = the average of the ICE curves).

THE DATA: hourly bike rentals with a KNOWN formula (ml_lab_utils.bike_components):
    rentals = 210 + hour x workday pattern + temperature hump + humidity drop + rain drop + noise
    wind_kmh and feels_like_c have NO effect, and feels_like_c is a near-copy of temp_c.
    So for once we can check an explanation against the truth.

LEVELS
    EASY   : Global (importance, PDP) vs local (what-if for one hour); PDP by hand = scikit-learn = truth?
    MEDIUM : ICE reveals what the PDP averages away (hour x workday); every feature's PDP vs the truth.
    HARD   : Correlated features fool the PDP (impossible combinations); ALE by hand, the fix;
             explanations stop at the edge of the data.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ml_lab_utils import BIKE_FEATURES, HAS_MPL, RANDOM_STATE, banner, bike_components, make_bike_rentals, note, plt, save_fig, show, step
from tree_toolkit import pdp_ice, permutation_importance_df

bikes = make_bike_rentals(n=10_000, seed=10, with_truth=True)
X, y = bikes[BIKE_FEATURES], bikes["rentals"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(X_tr, y_tr)
rmse = np.sqrt(mean_squared_error(y_te, model.predict(X_te)))
show("Test RMSE (rentals per hour) / noise sd in the formula", f"{rmse:.1f} / 25.0")


def true_effect(feature, grid):
    """The TRUE component curve for a feature (from the known formula), evaluated on a grid."""
    base = X_te.head(1)
    return bike_components(pd.concat([base] * len(grid)).assign(**{feature: grid}).reset_index(drop=True))[feature].to_numpy()


def centred(v):
    return np.asarray(v) - np.mean(v)


# %% =========================================================================
# LEVEL 1 (EASY): Global vs local
# ============================================================================
banner("Global: what drives the model? Local: why this prediction?", "EASY")
step("1.1  Global: permutation importance (test data)")
imp = permutation_importance_df(model, X_te, y_te,
                                lambda m, X_, y_: -np.sqrt(mean_squared_error(y_, m.predict(X_))), n_repeats=3)
print(imp.round(2).rename("RMSE increase when shuffled").to_string())

step("1.2  Global: partial dependence of temperature, by hand = scikit-learn = the truth?")
grid_t = np.linspace(0, 32, 17)
background = X_te.sample(500, random_state=0)
pdp_t, ice_t = pdp_ice(model.predict, background, "temp_c", grid_t)
sk = partial_dependence(model, background, ["temp_c"], custom_values={"temp_c": grid_t}, kind="average", method="brute")
truth_t = true_effect("temp_c", grid_t)
tab = pd.DataFrame({"temp_c": grid_t, "PDP by hand": pdp_t, "sklearn (brute)": sk["average"][0],
                    "PDP centred": centred(pdp_t), "TRUE effect centred": centred(truth_t)})
print(tab.iloc[::2].round(1).to_string(index=False))
show("Largest gap between centred PDP and the truth", f"{np.max(np.abs(centred(pdp_t) - centred(truth_t))):.1f} rentals")
note("""The PDP is a GLOBAL explanation: 'on average, how do predicted rentals change
with temperature?'. Our 3-line version equals scikit-learn's. Because we know
the real formula, we can check it: after centring (a PDP is defined up to a
constant), the model's temperature curve has the right hump around 24 C, but
it is FLATTER than the truth. Level 3 explains why (hint: feels_like_c).""")

step("1.3  Local: one specific hour, and what-if")
case = X_te.iloc[[7]]
print(case.to_string())
show("Prediction / actual rentals", f"{model.predict(case)[0]:.0f} / {y_te[7]:.0f}")
whatif = {"as is": case}
whatif["2 hours later"] = case.assign(hour=(case.hour + 2) % 24)
whatif["it starts raining"] = case.assign(rain=1)
whatif["10 C warmer"] = case.assign(temp_c=case.temp_c + 10, feels_like_c=case.feels_like_c + 10)
whatif["a workday instead" if case.workday.iloc[0] == 0 else "a weekend day instead"] = case.assign(workday=1 - case.workday)
print(pd.Series({k: model.predict(v)[0] for k, v in whatif.items()}).round(0).rename("predicted rentals").to_string())
note("""A LOCAL explanation answers questions about one case. 'What-if' (a
counterfactual) is the simplest: change one input, re-predict. It is easy to
understand, but only describes this model around this point. SHAP (Lab 11)
turns local explanations into an exact accounting of the prediction.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): What the average hides
# ============================================================================
banner("ICE curves and the PDP of every feature", "MEDIUM")
step("2.1  The PDP of hour averages two different worlds")
grid_h = np.arange(24)
pdp_h, ice_h = pdp_ice(model.predict, background, "hour", grid_h)
work = background.workday.to_numpy() == 1
truth_work = bike_components(pd.DataFrame({"hour": grid_h, "workday": 1, "temp_c": 20, "humidity": 50, "rain": 0}))["hour_x_workday"]
truth_free = bike_components(pd.DataFrame({"hour": grid_h, "workday": 0, "temp_c": 20, "humidity": 50, "rain": 0}))["hour_x_workday"]
hours = pd.DataFrame({"PDP (all rows)": pdp_h, "ICE mean: workdays": ice_h[work].mean(0), "ICE mean: weekends": ice_h[~work].mean(0),
                      "TRUE workday pattern": truth_work.to_numpy(), "TRUE weekend pattern": truth_free.to_numpy()}, index=grid_h)
print(hours.loc[[3, 8, 12, 14, 17, 20]].round(0).to_string())
note(f"""Split the ICE curves by workday and the truth appears: workdays have sharp
commuting peaks at 8:00 and 17:30, weekends a single broad afternoon hump. The
PDP averages the two: at 8:00 it says {hours.loc[8, 'PDP (all rows)']:.0f} (workdays {hours.loc[8, 'ICE mean: workdays']:.0f}, weekends {hours.loc[8, 'ICE mean: weekends']:.0f});
at 14:00 {hours.loc[14, 'PDP (all rows)']:.0f} (workdays {hours.loc[14, 'ICE mean: workdays']:.0f}, weekends {hours.loc[14, 'ICE mean: weekends']:.0f}). It describes NO real day. Whenever ICE curves fan out into
families, there is an INTERACTION, and the PDP alone misleads: show ICE, or a
2-D PDP (hour x workday).""")

step("2.2  Every feature's PDP vs the truth")
rows = {}
for f, grid in {"temp_c": grid_t, "humidity": np.linspace(30, 95, 14), "rain": np.array([0, 1]),
                "wind_kmh": np.linspace(0, 35, 8), "feels_like_c": grid_t}.items():
    pdp_f, _ = pdp_ice(model.predict, background, f, grid)
    truth = true_effect(f, grid) if f in ("temp_c", "humidity", "rain") else np.zeros(len(grid))
    rows[f] = {"PDP range (max - min)": np.ptp(pdp_f), "TRUE range": np.ptp(truth),
               "max |PDP - truth| (centred)": np.max(np.abs(centred(pdp_f) - centred(truth)))}
pdp_tab = pd.DataFrame(rows).T
print(pdp_tab.round(1).to_string())
note(f"""Humidity and rain are recovered well, and wind (truly no effect) is flat. But
look at feels_like_c: its TRUE effect is zero, yet its PDP spans
{pdp_tab.loc['feels_like_c', 'PDP range (max - min)']:.0f} rentals, while temp_c falls {pdp_tab.loc['temp_c', 'TRUE range'] - pdp_tab.loc['temp_c', 'PDP range (max - min)']:.0f} short of its true range. The model
split the temperature effect between two near-identical columns, and each PDP
shows only its share.""")

# %% =========================================================================
# LEVEL 3 (HARD): Correlated features, ALE, and the edge of the data
# ============================================================================
banner("When partial dependence misleads, and a fix", "HARD")
step("3.1  The PDP asks about impossible days")
ice_rows = background.copy()
show("Correlation temp_c / feels_like_c", f"{X.temp_c.corr(X.feels_like_c):.3f}")
gap = np.abs(np.subtract.outer(grid_t, background.feels_like_c.to_numpy()))   # set temp to v, keep each row's feels-like
show("Share of PDP evaluations with |temp - feels_like| > 10 C", f"{(gap > 10).mean():.0%}")
show("...such days in the real data", f"{(np.abs(X.temp_c - X.feels_like_c) > 10).mean():.1%}")
note("""To compute PDP(temp = 30), every row gets temp 30 while KEEPING its own
feels-like temperature, including -5 C. Days like that never happen, so the
model was never trained on them; a large share of the PDP's evaluations are
in this no-data zone, and the result mixes real behaviour with arbitrary
extrapolation. With correlated features, the PDP answers a question about a
world that doesn't exist.""")

step("3.2  Accumulated Local Effects (ALE): only compare like with like")


def ale(predict, X_, feature, n_bins=16):
    """ALE (Apley & Zhu, 2020): within each quantile bin of the feature, move ONLY the rows that are
    actually in that bin from the bin's lower to its upper edge, average the change in prediction,
    then accumulate the averages and centre. Rows are never moved far from where they really are."""
    edges = np.unique(np.quantile(X_[feature], np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, X_[feature], side="right") - 1, 0, len(edges) - 2)
    effects = []
    for k in range(len(edges) - 1):
        rows_k = X_[idx == k]
        if len(rows_k) == 0:
            effects.append(0.0)
            continue
        effects.append(np.mean(predict(rows_k.assign(**{feature: edges[k + 1]})) - predict(rows_k.assign(**{feature: edges[k]}))))
    curve = np.r_[0, np.cumsum(effects)]
    weights = np.bincount(idx, minlength=len(edges) - 1)
    mid = (curve[1:] + curve[:-1]) / 2
    return edges, curve - np.sum(mid * weights) / weights.sum()


edges, ale_t = ale(model.predict, X_te, "temp_c")
pdp_e, _ = pdp_ice(model.predict, background, "temp_c", edges)
truth_e = true_effect("temp_c", edges)
in_range = (edges >= 0) & (edges <= 32)
cmp_ = pd.DataFrame({"temp_c": edges, "PDP (centred)": centred(pdp_e), "ALE": ale_t, "TRUE (centred)": centred(truth_e)})
print(cmp_.iloc[::2].round(1).to_string(index=False))
err = {"PDP": np.mean(np.abs(centred(pdp_e)[in_range] - centred(truth_e)[in_range])),
       "ALE": np.mean(np.abs((ale_t - ale_t[in_range].mean())[in_range] - centred(truth_e[in_range])))}
show("Mean |curve - truth| over 0-32 C: PDP / ALE", f"{err['PDP']:.1f} / {err['ALE']:.1f} rentals")
no_feels = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(
    X_tr.drop(columns="feels_like_c"), y_tr)
pdp_nf, _ = pdp_ice(no_feels.predict, background.drop(columns="feels_like_c"), "temp_c", grid_t)
show("Model WITHOUT feels_like_c: max |PDP - truth|", f"{np.max(np.abs(centred(pdp_nf) - centred(truth_t))):.1f} rentals")
note(f"""ALE never moves a row far from where it really is: within each narrow band of
temperature it nudges only the rows that are in that band (whose feels-like
temperature matches), and adds up those local changes. It is closer to the
truth than the PDP ({err['ALE']:.1f} vs {err['PDP']:.1f} rentals on average), most clearly at the hot edge
(38 C: ALE {cmp_.ALE.iloc[-1]:.0f}, PDP {cmp_['PDP (centred)'].iloc[-1]:.0f}, truth {cmp_['TRUE (centred)'].iloc[-1]:.0f}), where the PDP drags cold feels-like
values into hot days. Both still see only temp_c's share of an effect the
model split across twins; removing the redundant column helps too (max gap
{np.max(np.abs(centred(pdp_t) - centred(truth_t))):.0f} -> {np.max(np.abs(centred(pdp_nf) - centred(truth_t))):.0f}). Explain correlated features as a GROUP, or remove duplicates first.""")

step("3.3  Explanations stop at the edge of the data")
far = np.array([30, 34, 38, 42, 46])
pdp_far, _ = pdp_ice(no_feels.predict, background.drop(columns="feels_like_c"), "temp_c", far)
print(pd.DataFrame({"temp_c": far, "PDP": pdp_far.round(0), "TRUE effect": true_effect("temp_c", far).round(0),
                    "training rows at this temperature or hotter": [(X_tr.temp_c >= t).sum() for t in far]}).to_string(index=False))
note("""Above ~38 C there are no training days. Trees can't extrapolate (Module 3,
Lab 07): the model, and therefore every explanation of it, just repeats the
edge of the data, while the true effect keeps falling. An explanation is
only as trustworthy as the data under it: always show where the data is
(rug plots, histograms) next to a PDP or ALE curve.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for i in range(0, 120, 3):
        axes[0].plot(grid_h, ice_h[i], color="#e45756" if work[i] else "#4c78a8", alpha=0.25, lw=0.8)
    axes[0].plot(grid_h, pdp_h, "k-", lw=2.5, label="PDP (average of all curves)")
    axes[0].set(xlabel="hour", ylabel="predicted rentals", title="ICE: red = workdays, blue = weekends")
    axes[0].legend()
    axes[1].plot(grid_t, centred(truth_t), "k--", lw=2, label="TRUE effect")
    axes[1].plot(grid_t, centred(pdp_t), label="PDP (with feels_like_c)")
    axes[1].plot(edges, ale_t - ale_t[in_range].mean(), label="ALE (with feels_like_c)")
    axes[1].plot(grid_t, centred(pdp_nf), label="PDP (feels_like_c removed)")
    axes[1].set(xlabel="temperature (C)", ylabel="effect on rentals (centred)", title="Temperature: explanations vs truth")
    axes[1].legend(fontsize=8)
    axes[2].scatter(X.temp_c, X.feels_like_c, s=2, alpha=0.3)
    axes[2].scatter([30] * 3, [-5, 5, 15], color="r", s=40, label="points a PDP evaluates")
    axes[2].set(xlabel="temp_c", ylabel="feels_like_c", title="Correlated features: PDP leaves the data")
    axes[2].legend()
    save_fig(fig, "10_pdp_ice_ale")

banner("KEY TAKEAWAYS")
print("""  1. Global explanations (importance, PDP, ALE) describe the model overall; local ones explain one case.
  2. PDP = average prediction when everyone gets value v; ICE = the same curve per row.
  3. ICE curves that fan out reveal interactions the PDP averages away: show ICE or a 2-D PDP.
  4. With correlated features PDP evaluates impossible cases; ALE stays local; better, remove duplicates.
  5. No explanation is valid beyond the data the model was trained on.""")
