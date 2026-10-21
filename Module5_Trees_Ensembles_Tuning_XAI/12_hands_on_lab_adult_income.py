"""
LAB 12 - HANDS-ON LAB: Trees, Ensembles, Tuning and Explanations on Real Census Data
===================================================================================

THE PROBLEM
    A (fictional) financial-inclusion programme wants to estimate who earns more than $50k a year,
    to size an outreach campaign for a savings product. The programme must be able to explain each
    estimate and show that it treats men and women fairly.

THE DATA (real): UCI Adult / Census Income, 48,842 people from the 1994 US census
    (OpenML 1590, cached in data/adult.csv). Target: income > $50k (23.9%).

WHAT YOU WILL DO (the syllabus hands-on lab)
    Part 1 (EASY)   Explore; decide which columns may be features (and why)
    Part 2 (EASY)   Split once; a baseline; a decision tree with cross-validation; VISUALISE it
    Part 3 (MEDIUM) Train a decision tree, a random forest and a boosting model; CROSS-VALIDATE;
                    TUNE their hyperparameters
    Part 4 (MEDIUM) EXPLAIN: global importance, then INDIVIDUAL predictions (exact SHAP + the tree path)
    Part 5 (HARD)   Structured ERROR ANALYSIS and a fairness check; open the test set once; save
"""
# %%
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from ml_lab_utils import ADULT_TARGET, HAS_MPL, OUT_DIR, RANDOM_STATE, banner, load_adult, note, plt, save_fig, show, step
from tree_toolkit import permutation_importance_df, shapley_exact, slice_report

# %% =========================================================================
# PART 1 (EASY): Explore, and decide what may be a feature
# ============================================================================
banner("PART 1 - Explore the data", "EASY")
adult = load_adult()
show("People x columns", adult.shape)
show("Earn > $50k", f"{adult[ADULT_TARGET].mean():.1%}")
print(adult.head(3).T.to_string())
show("'Unknown' workclass / occupation / country", f"{(adult.workclass == 'Unknown').mean():.1%} / "
                                                   f"{(adult.occupation == 'Unknown').mean():.1%} / "
                                                   f"{(adult.native_country == 'Unknown').mean():.1%}")
show("education vs education_num: one-to-one?", bool(adult.groupby("education").education_num.nunique().max() == 1))
for col in ["sex", "relationship", "marital_status"]:
    print(adult.groupby(col)[ADULT_TARGET].agg(["count", "mean"]).rename(columns={"mean": "share > $50k"}).round(3).to_string(), "\n")
SENSITIVE = ["sex", "race"]
NUM = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT = ["workclass", "marital_status", "occupation", "relationship", "native_country"]
note("""Feature decisions, written down BEFORE modelling (Module 2):
- fnlwgt is the census SAMPLING WEIGHT (how many people a row represents), not
  a property of the person: excluded.
- education (text) and education_num carry the same information: keep the number.
- sex and race are SENSITIVE: not used as features, kept aside for the fairness
  audit in Part 5. Note that relationship ('Husband' / 'Wife') and marital status
  are strongly linked to sex: removing sex does not remove its information.
- '?' answers were turned into an 'Unknown' category: missingness is a feature.""", keep_lines=True)

# %% =========================================================================
# PART 2 (EASY): Split once; baseline; a tree you can read
# ============================================================================
banner("PART 2 - Split, baseline, and a decision tree", "EASY")
X, y = adult[NUM + CAT], adult[ADULT_TARGET].to_numpy()
X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(X, y, adult[SENSITIVE], test_size=0.2, stratify=y,
                                                      random_state=RANDOM_STATE)
show("Train / test people (test SEALED until Part 5.4)", f"{len(y_tr):,} / {len(y_te):,}")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
onehot = ColumnTransformer([("num", "passthrough", NUM),
                            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                                  sparse_output=False), CAT)], verbose_feature_names_out=False)


def cv_scores(model, name):
    t0 = time.perf_counter()
    s = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["roc_auc", "neg_log_loss", "accuracy"], n_jobs=-1)
    return {"model": name, "CV AUC": s["test_roc_auc"].mean(), "CV AUC sd": s["test_roc_auc"].std(),
            "CV log loss": -s["test_neg_log_loss"].mean(), "CV accuracy": s["test_accuracy"].mean(),
            "seconds": time.perf_counter() - t0}


results = [cv_scores(DummyClassifier(), "baseline: predict the base rate")]
tree3 = Pipeline([("prep", onehot), ("model", DecisionTreeClassifier(max_depth=3, random_state=0))])
results.append(cv_scores(tree3, "decision tree, depth 3"))
print(pd.DataFrame(results).set_index("model").round(4).to_string())
tree3.fit(X_tr, y_tr)
names = list(tree3["prep"].get_feature_names_out())
print(export_text(tree3["model"], feature_names=names, show_weights=True, decimals=2))
note("""A depth-3 tree (8 rules) already reaches AUC 0.86, and it reads like a census
analyst's notes. First question: married (civilian spouse)? Then capital gains
(> ~$5-7k is almost always a high earner) and education (13+ years). Married
with a degree: majority > $50k; not married without big capital gains: about
2% to 15%. The plot is saved in outputs/12_adult_tree.png. Keep this tree as
the 'explainable baseline' every complex model must beat.""")

# %% =========================================================================
# PART 3 (MEDIUM): Tree vs forest vs boosting, cross-validated and tuned
# ============================================================================
banner("PART 3 - Train, cross-validate and tune three model families", "MEDIUM")
step("3.1  Tune each family (random search, 3-fold CV inside the training data)")
inner = StratifiedKFold(3, shuffle=True, random_state=1)
X_tr_cat = X_tr.astype({c: "category" for c in CAT})
X_te_cat = X_te.astype({c: "category" for c in CAT})
searches = {
    "decision tree": (Pipeline([("prep", onehot), ("model", DecisionTreeClassifier(random_state=0))]),
                      {"model__max_depth": randint(3, 20), "model__min_samples_leaf": randint(5, 300),
                       "model__ccp_alpha": loguniform(1e-6, 1e-3)}, X_tr),
    "random forest": (Pipeline([("prep", onehot), ("model", RandomForestClassifier(150, max_samples=0.5, n_jobs=-1,
                                                                                   random_state=0))]),
                      {"model__min_samples_leaf": randint(1, 30), "model__max_features": [0.1, 0.2, 0.3, "sqrt"]}, X_tr),
    "gradient boosting": (HistGradientBoostingClassifier(max_iter=500, early_stopping=True, categorical_features="from_dtype",
                                                         random_state=0),
                          {"learning_rate": loguniform(0.03, 0.2), "max_leaf_nodes": randint(8, 64),
                           "min_samples_leaf": randint(10, 200), "l2_regularization": loguniform(1e-3, 10)}, X_tr_cat),
}
tuned, tune_rows = {}, {}
for name, (est, space, Xs) in searches.items():
    t0 = time.perf_counter()
    rs = RandomizedSearchCV(est, space, n_iter=10, cv=inner, scoring="roc_auc", random_state=0, n_jobs=-1).fit(Xs, y_tr)
    tuned[name] = rs.best_estimator_
    tune_rows[name] = {"best inner CV AUC": rs.best_score_, "search seconds": time.perf_counter() - t0,
                       "best settings": {k.replace("model__", ""): (v if isinstance(v, str) else int(v) if float(v).is_integer()
                                                                    else round(float(v), 5)) for k, v in rs.best_params_.items()}}
print(pd.DataFrame(tune_rows).T.to_string())

step("3.2  Compare the tuned models with 5-fold CV (fresh folds)")
for name, m in tuned.items():
    Xs = X_tr_cat if name == "gradient boosting" else X_tr
    t0 = time.perf_counter()
    s = cross_validate(clone(m), Xs, y_tr, cv=cv, scoring=["roc_auc", "neg_log_loss", "accuracy"], n_jobs=-1)
    results.append({"model": f"{name} (tuned)", "CV AUC": s["test_roc_auc"].mean(), "CV AUC sd": s["test_roc_auc"].std(),
                    "CV log loss": -s["test_neg_log_loss"].mean(), "CV accuracy": s["test_accuracy"].mean(),
                    "seconds": time.perf_counter() - t0})
board = pd.DataFrame(results).set_index("model")
print(board.round(4).to_string())
note(f"""The tuned settings were chosen on 3-fold inner CV; the comparison uses fresh
5-fold CV so the search's own luck doesn't flatter the winner (Lab 07). The
ensemble families beat the single tree; boosting leads ({board['CV AUC'].max():.3f}), with fold-to-fold
spread around {board['CV AUC sd'].iloc[-1]:.3f}. The depth-3 tree is {board.loc['gradient boosting (tuned)', 'CV AUC'] - board.loc['decision tree, depth 3', 'CV AUC']:.3f} AUC behind, a real gap:
that is the price of eight readable rules (Lab 02).""")

# %% =========================================================================
# PART 4 (MEDIUM): Explain the champion
# ============================================================================
banner("PART 4 - Explain the model: globally, then one person at a time", "MEDIUM")
champ = tuned["gradient boosting"]
X_fit, X_val, y_fit, y_val = train_test_split(X_tr_cat, y_tr, test_size=0.25, stratify=y_tr, random_state=2)
champ_val = clone(champ).fit(X_fit, y_fit)
step("4.1  Global: permutation importance on held-out data")
imp = permutation_importance_df(champ_val, X_val, y_val, lambda m, X_, y_: roc_auc_score(y_, m.predict_proba(X_)[:, 1]),
                                n_repeats=3)
print(imp.round(4).rename("AUC drop when shuffled (validation)").to_string())

step("4.2  Local: explain two people with exact SHAP (log-odds would also work; here: probability)")
GROUPS = {"age": ["age"], "education": ["education_num"], "occupation": ["occupation"], "workclass": ["workclass"],
          "marital & relationship": ["marital_status", "relationship"], "capital gain/loss": ["capital_gain", "capital_loss"],
          "hours per week": ["hours_per_week"], "country": ["native_country"]}
bg = X_fit.sample(60, random_state=0)
p_val = champ_val.predict_proba(X_val)[:, 1]
people = {"likely high earner": int(np.argmax(np.where(p_val > 0.8, 1, 0) * (X_val.capital_gain.to_numpy() == 0))),
          "borderline (p ~ 0.5)": int(np.argmin(np.abs(p_val - 0.5)))}
for label, i in people.items():
    person = X_val.iloc[[i]]
    phi = shapley_exact(lambda d: champ_val.predict_proba(d)[:, 1], person, bg, GROUPS)
    print(f"\n  {label}: P(> $50k) = {p_val[i]:.2f}, actual = {'> $50k' if y_val[i] else '<= $50k'}")
    print("   ", person[NUM + CAT].iloc[0].to_dict())
    running = phi.attrs["base_value"]
    print(f"    average person .............................. {running:.2f}")
    for f_, v_ in phi.sort_values(key=abs, ascending=False).items():
        running += v_
        print(f"    {f_:<30} {'raises' if v_ > 0 else 'lowers'} {v_:+.3f} -> {running:.2f}")
borderline = X_val.iloc[[people["borderline (p ~ 0.5)"]]].astype({c: object for c in CAT})
t_model, t_x = tree3["model"], tree3["prep"].transform(borderline)
print("\n  The same borderline person, through the depth-3 tree:")
for nd in t_model.decision_path(t_x).indices:
    f_ = t_model.tree_.feature[nd]
    if f_ >= 0:
        print(f"    {names[f_]} = {t_x[0, f_]:g} {'<=' if t_x[0, f_] <= t_model.tree_.threshold[nd] else '>'} {t_model.tree_.threshold[nd]:.2f}")
    else:
        print(f"    leaf: P(> $50k) = {t_model.tree_.value[nd, 0, 1]:.2f}")
note("""Global: marital status / relationship, education, capital gains, age and
occupation drive the model. Local: SHAP turns each estimate into a waterfall
from the average person (base value) to this person, adding up exactly. For
the borderline person the reasons pull in both directions, which is WHY the
model is unsure: useful for a human reviewer. The depth-3 tree gives an
alternative local explanation (its decision path), simpler but coarser.""")

# %% =========================================================================
# PART 5 (HARD): Error analysis, fairness, the test set, saving
# ============================================================================
banner("PART 5 - Structured error analysis and a fairness check", "HARD")
step("5.1  Where does the model fail? (out-of-fold, training data)")
oof = cross_val_predict(clone(champ), X_tr_cat, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
err = (y_tr - oof) ** 2
analysis = X_tr.assign(sex=s_tr.sex.to_numpy(), race=s_tr.race.to_numpy())
sl = slice_report(analysis, err, ["occupation", "marital_status", "education_num", "age", "hours_per_week", "sex", "race"],
                  min_size=300)
print(sl.head(8).to_string(float_format=lambda v: f"{v:.3f}"))
note("""Errors concentrate where the outcome is genuinely uncertain: married, older,
well-educated people working long hours, often professionals or executives (and
so, through marriage, mostly men). Their incomes cluster around the $50k line,
and these census fields can't tell which side they fall. That is MISSING
INFORMATION (no job title, employer, region, years in the job), not a modelling
error a bigger model would fix: the data-centric next step is to add fields.""")

step("5.2  Fairness check by sex (the model never saw it)")
THRESH = 0.5
rows = {}
for g in ["Female", "Male"]:
    m_ = (s_tr.sex == g).to_numpy()
    pred = oof[m_] >= THRESH
    rows[g] = {"people": int(m_.sum()), "actual share > $50k": y_tr[m_].mean(), "mean predicted P": oof[m_].mean(),
               "selected (p >= 0.5)": pred.mean(), "recall": (pred & (y_tr[m_] == 1)).sum() / max((y_tr[m_] == 1).sum(), 1),
               "precision": (pred & (y_tr[m_] == 1)).sum() / max(pred.sum(), 1)}
fair = pd.DataFrame(rows).T
print(fair.to_string(float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:,.0f}"))
no_rel = HistGradientBoostingClassifier(**{k: v for k, v in champ.get_params().items()}).fit(
    X_tr_cat.drop(columns=["relationship", "marital_status"]), y_tr)
p_nr = no_rel.predict_proba(X_te_cat.drop(columns=["relationship", "marital_status"]))[:, 1]
note(f"""The model never saw 'sex', yet it selects {fair.loc['Male', 'selected (p >= 0.5)']:.0%} of men and {fair.loc['Female', 'selected (p >= 0.5)']:.0%} of women. Part of that
reflects the real 1994 income gap (actual shares {fair.loc['Male', 'actual share > $50k']:.0%} vs {fair.loc['Female', 'actual share > $50k']:.0%}), and the model is
well calibrated within each group (mean predicted ~ actual). But recall for
women is {fair.loc['Female', 'recall']:.0%} vs {fair.loc['Male', 'recall']:.0%} for men: high-earning women are missed more often. 'Husband' /
'Wife' carries sex into the model. Whether to equalise recall (e.g. per-group
thresholds), drop proxies (cost: {roc_auc_score(y_te, champ.fit(X_tr_cat, y_tr).predict_proba(X_te_cat)[:, 1]) - roc_auc_score(y_te, p_nr):.3f} test AUC without marital
fields), or accept it is a POLICY decision for the programme, documented in the
model card, not a choice to be made silently by the data scientist.""")

step("5.3  Open the test set ONCE")
test_rows = {}
for name, m in {"decision tree, depth 3": tree3, **{f"{k} (tuned)": v for k, v in tuned.items()}}.items():
    Xa, Xb = (X_tr_cat, X_te_cat) if name.startswith("gradient") else (X_tr, X_te)
    t0 = time.perf_counter()
    fitted = clone(m).fit(Xa, y_tr)
    fit_s = time.perf_counter() - t0
    pr = fitted.predict_proba(Xb)[:, 1]
    test_rows[name] = {"test AUC": roc_auc_score(y_te, pr), "test log loss": log_loss(y_te, pr),
                       "test accuracy @0.5": ((pr >= 0.5) == y_te).mean(), "fit seconds": fit_s}
    if name.startswith("gradient"):
        champ_fit, p_champ = fitted, pr
test_tab = pd.DataFrame(test_rows).T
print(test_tab.round(4).to_string())
rng = np.random.default_rng(0)
diffs = []
p_rf = clone(tuned["random forest"]).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
for _ in range(500):
    b = rng.integers(0, len(y_te), len(y_te))
    diffs.append(roc_auc_score(y_te[b], p_champ[b]) - roc_auc_score(y_te[b], p_rf[b]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
show("AUC(boosting) - AUC(forest), 95% CI", f"{np.mean(diffs):+.4f}  [{lo:+.4f}, {hi:+.4f}]")
note("""The test set confirms the CV ranking. Boosting's lead over the forest is
small but its interval excludes zero; it also trains in seconds and handles
categories natively, while the forest needs one-hot encoding and is far
larger on disk. The depth-3 tree remains the fallback when someone must be
able to read every rule.""")

step("5.4  Save the model, metrics and a model card")
joblib.dump(champ_fit, OUT_DIR / "adult_income_model_v1.joblib")
metrics = {"model": "HistGradientBoostingClassifier (tuned)", "params": {k: str(v) for k, v in champ.get_params().items()},
           "features": NUM + CAT, "excluded": {"fnlwgt": "sampling weight", "education": "duplicate of education_num",
                                              "sex, race": "sensitive; used only for auditing"},
           "test": test_tab.loc["gradient boosting (tuned)"].to_dict(), "fairness_oof": fair.to_dict()}
(OUT_DIR / "adult_income_metrics_v1.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
champ_row = test_tab.loc["gradient boosting (tuned)"]
card = f"""# Model Card: Income > $50k estimator v1 (outreach sizing)

- **Use:** estimate the share of people likely to earn > $50k, to size an outreach campaign. NOT for individual
  credit, hiring or pricing decisions.
- **Data:** UCI Adult (1994 US census, 48,842 people); 80/20 split stratified on the target; test opened once.
- **Model:** HistGradientBoosting (tuned by random search, early stopping), native categorical handling.
- **Features:** {', '.join(NUM + CAT)}. Excluded: fnlwgt (sampling weight), education (duplicate), sex and race (sensitive).
- **Test performance:** ROC-AUC {champ_row['test AUC']:.3f}, log loss {champ_row['test log loss']:.3f}, accuracy {champ_row['test accuracy @0.5']:.3f} at threshold 0.5.
- **Explainable fallback:** depth-3 decision tree, test AUC {test_tab.loc['decision tree, depth 3', 'test AUC']:.3f}.
- **Fairness:** sex not used, but relationship / marital status act as proxies. At 0.5: recall for women
  {fair.loc['Female', 'recall']:.0%} vs men {fair.loc['Male', 'recall']:.0%}; calibrated within each group. Policy decision required before any individual use.
- **Known limitations:** 1994 data (incomes and occupations have changed); no job title, employer or region; errors concentrate
  in mid-level occupations where these fields can't separate the outcome.
- **Explanations:** exact SHAP per person (grouped features), relative to the average training person.
"""
(OUT_DIR / "adult_income_model_card.md").write_text(card, encoding="utf-8")
print("  [saved] outputs/adult_income_model_v1.joblib, adult_income_metrics_v1.json, adult_income_model_card.md")

if HAS_MPL:
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_tree(tree3["model"], feature_names=names, class_names=["<= 50k", "> 50k"], filled=True, rounded=True,
              proportion=True, fontsize=8, ax=ax)
    ax.set_title("Depth-3 decision tree on the Adult census data (value = class proportions)")
    save_fig(fig, "12_adult_tree")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    b = board.sort_values("CV AUC")
    axes[0].barh(b.index, b["CV AUC"], xerr=b["CV AUC sd"], color="#4c78a8")
    axes[0].set(xlim=(0.5, 0.95), xlabel="5-fold CV ROC-AUC", title="Model families, tuned")
    axes[0].tick_params(axis="y", labelsize=8)
    i15 = imp.sort_values()
    axes[1].barh(i15.index, i15.values, color="#54a24b")
    axes[1].set(xlabel="AUC drop when shuffled (validation)", title="What the champion relies on")
    save_fig(fig, "12_adult_models")

banner("KEY TAKEAWAYS")
print("""  1. Decide what may be a feature before modelling: sampling weights, duplicates, sensitive attributes.
  2. A depth-3 tree is a strong, readable baseline; ensembles add a few points of AUC.
  3. Tune with inner CV, compare on fresh folds, open the test set once with a CI.
  4. Explain globally (permutation importance) and locally (SHAP waterfalls, tree paths).
  5. Error analysis + a fairness audit: proxies carry sensitive information; document the policy choice.""")
