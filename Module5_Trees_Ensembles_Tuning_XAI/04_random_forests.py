"""
LAB 04 - Random Forests in Depth
================================

THE RECIPE (Breiman, 2001)
    repeat B times:
        draw a bootstrap sample of the rows (same size, with replacement: ~63% unique rows)
        grow a DEEP tree; at every split consider only a random subset of max_features features
    predict: average the trees' probabilities (classification) or values (regression)

    The ~37% of rows a tree never saw are its OUT-OF-BAG (OOB) rows: scoring each row only with the
    trees that didn't see it gives a free, honest validation estimate.

THE KNOBS THAT MATTER (in order)
    max_features      decorrelation vs tree strength (Lab 03, 3.2)        default sqrt(p)
    min_samples_leaf  leaf size: smoother probabilities, smaller model    default 1
    n_estimators      more = more stable, never worse, slower             default 100
    max_samples       rows per bootstrap: faster, more diverse trees      default all
    (ExtraTrees: also randomise the THRESHOLDS -> even more decorrelated, faster to train)

LEVELS
    EASY   : A forest in three lines; out-of-bag vs cross-validation vs test.
    MEDIUM : Each knob's effect on accuracy and fit time; ExtraTrees; class_weight.
    HARD   : OOB fooled by duplicate rows; what a forest costs to deploy (size, latency);
             forest probabilities vs the TRUE probabilities, and calibration.
"""
# %%
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from ml_lab_utils import CREDIT_NUM, HAS_MPL, RANDOM_STATE, banner, make_credit_applicants, note, plt, save_fig, show, step

credit = make_credit_applicants(n=12_000, seed=4, with_truth=True)
X = pd.get_dummies(credit[CREDIT_NUM + ["employment"]], columns=["employment"], dtype=int)
y, p_true = credit["default"].to_numpy(), credit["_true_prob"].to_numpy()
X_tr, X_te, y_tr, y_te, pt_tr, pt_te = train_test_split(X, y, p_true, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
show("Train / test applicants", f"{len(y_tr):,} / {len(y_te):,}")
show("Best possible test AUC (true probabilities)", f"{roc_auc_score(y_te, pt_te):.3f}")


def oob_auc(rf):
    return roc_auc_score(y_tr, rf.oob_decision_function_[:, 1])


# %% =========================================================================
# LEVEL 1 (EASY): A forest, and its free validation score
# ============================================================================
banner("A random forest in three lines", "EASY")
rf = RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1, random_state=0)
rf.fit(X_tr, y_tr)
p_rf = rf.predict_proba(X_te)[:, 1]
tree = DecisionTreeClassifier(min_samples_leaf=50, random_state=0).fit(X_tr, y_tr)
rows = {"single tuned tree (leaf >= 50)": {"test AUC": roc_auc_score(y_te, tree.predict_proba(X_te)[:, 1])},
        "random forest (300 trees)": {"test AUC": roc_auc_score(y_te, p_rf)}}
print(pd.DataFrame(rows).T.round(3).to_string())
in_bag = np.zeros(len(y_tr))
for est, samp in zip(rf.estimators_, rf.estimators_samples_):
    in_bag[np.unique(samp)] += 1
show("Average share of trees that saw a given row", f"{in_bag.mean() / len(rf.estimators_):.3f}  (theory: 1 - 1/e = 0.632)")
show("Out-of-bag AUC (free)", f"{oob_auc(rf):.3f}")
show("5-fold CV AUC (5 extra fits)", f"{cross_val_score(RandomForestClassifier(300, n_jobs=-1, random_state=0), X_tr, y_tr, cv=cv, scoring='roc_auc').mean():.3f}")
show("Test AUC (the truth we normally don't have)", f"{roc_auc_score(y_te, p_rf):.3f}")
note("""Each tree sees ~63% of the training rows. Scoring every row with only the
~110 trees that did NOT see it gives the out-of-bag estimate: as honest as
cross-validation, but free (no refitting). Use oob_score=True to tune a forest
quickly; switch to CV when you compare it with other model families.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): The knobs
# ============================================================================
banner("What each hyperparameter does", "MEDIUM")


def run(name, model):
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    return {"OOB AUC": oob_auc(model), "fit seconds": time.perf_counter() - t0, "_m": model}


step("2.1  n_estimators: more trees, a smoother OOB curve, never worse")
rows = {n: run(n, RandomForestClassifier(n, oob_score=True, n_jobs=-1, random_state=0)) for n in [50, 100, 300, 600, 1000]}
n_tab = pd.DataFrame(rows).T.drop(columns="_m")
n_tab.index.name = "n_estimators"
print(n_tab.astype(float).round(4).to_string())
note(f"""OOB AUC rises from {n_tab['OOB AUC'].iloc[0]:.3f} to {n_tab['OOB AUC'].iloc[-1]:.3f} and flattens: extra trees only make the average
more precise. (With few trees each row is out-of-bag for only a handful of
them, which makes the OOB estimate itself pessimistic.) Time grows linearly:
stop adding trees when the curve is flat.""")

step("2.2  max_features, min_samples_leaf, max_samples, ExtraTrees")
grid = {
    "default (sqrt, leaf 1)": RandomForestClassifier(300, oob_score=True, n_jobs=-1, random_state=0),
    "max_features = 0.6": RandomForestClassifier(300, max_features=0.6, oob_score=True, n_jobs=-1, random_state=0),
    "max_features = None (bagging)": RandomForestClassifier(300, max_features=None, oob_score=True, n_jobs=-1, random_state=0),
    "min_samples_leaf = 10": RandomForestClassifier(300, min_samples_leaf=10, oob_score=True, n_jobs=-1, random_state=0),
    "min_samples_leaf = 30": RandomForestClassifier(300, min_samples_leaf=30, oob_score=True, n_jobs=-1, random_state=0),
    "max_samples = 0.3": RandomForestClassifier(300, max_samples=0.3, oob_score=True, n_jobs=-1, random_state=0),
    "leaf 10 + max_samples 0.5": RandomForestClassifier(300, min_samples_leaf=10, max_samples=0.5, oob_score=True,
                                                        n_jobs=-1, random_state=0),
    "ExtraTrees, leaf 10": ExtraTreesClassifier(300, min_samples_leaf=10, bootstrap=True, oob_score=True, n_jobs=-1,
                                                random_state=0),
}
rows = {name: run(name, m) for name, m in grid.items()}
knob = pd.DataFrame(rows).T
knob["test AUC"] = [roc_auc_score(y_te, r["_m"].predict_proba(X_te)[:, 1]) for r in rows.values()]
knob = knob.drop(columns="_m").astype(float).sort_values("OOB AUC", ascending=False)
print(knob.round(4).to_string())
note(f"""On this data (8,400 training applicants, a smooth true probability) the biggest
lever is min_samples_leaf: leaves of 10-30 rows average out the label noise
that a leaf of 1 memorises. Smaller bootstrap samples (max_samples) add
diversity and, on large data, save time. ExtraTrees (random thresholds) is
competitive. Using all features at every split ('bagging') is the weakest:
correlated trees (Lab 03). OOB and test AUC rank the settings almost the same
way, so OOB is a safe and cheap tuning signal.""")

step("2.3  class_weight='balanced_subsample': a threshold shift, again")
rows = {}
for cw in [None, "balanced_subsample"]:
    m = RandomForestClassifier(300, min_samples_leaf=10, class_weight=cw, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    pr = m.predict_proba(X_te)[:, 1]
    rows[str(cw)] = {"test AUC": roc_auc_score(y_te, pr), "mean P(default)": pr.mean(), "actual rate": y_te.mean(),
                     "recall @0.5": ((pr >= 0.5) & (y_te == 1)).sum() / y_te.sum()}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""As in Module 4 (Lab 05): class weights barely change the RANKING (AUC); they
inflate the probabilities, which raises recall at 0.5. Choose the threshold
instead, and keep honest probabilities.""")

# %% =========================================================================
# LEVEL 3 (HARD): OOB traps, deployment cost, probabilities
# ============================================================================
banner("Duplicates, deployment cost and calibration", "HARD")

step("3.1  OOB is fooled by duplicate rows")
dup = X_tr.sample(frac=0.5, random_state=1)
X_dup = pd.concat([X_tr, dup])
y_dup = np.r_[y_tr, y_tr[X_tr.index.get_indexer(dup.index)]]
m_dup = RandomForestClassifier(300, oob_score=True, n_jobs=-1, random_state=0).fit(X_dup, y_dup)
rows = {"clean training data": {"OOB AUC": oob_auc(rf), "test AUC": roc_auc_score(y_te, p_rf)},
        "50% of rows duplicated": {"OOB AUC": roc_auc_score(y_dup, m_dup.oob_decision_function_[:, 1]),
                                   "test AUC": roc_auc_score(y_te, m_dup.predict_proba(X_te)[:, 1])}}
oob_tab = pd.DataFrame(rows).T
print(oob_tab.round(3).to_string())
note(f"""A duplicated row can be out-of-bag for a tree that saw its twin: the 'unseen'
row is not unseen. With a third of the rows duplicated, OOB AUC jumps to
{oob_tab.iloc[1, 0]:.3f} while the true (test) AUC doesn't move. Duplicates come from double
loads, joins, or oversampling (Module 4, Lab 05). De-duplicate before
trusting OOB, and when rows come in groups (the same customer many times),
use GROUP cross-validation instead (Lab 06).""")

step("3.2  What a forest costs to deploy")
rows = {}
for name, m in {"default: 300 trees, leaf 1": RandomForestClassifier(300, n_jobs=-1, random_state=0),
                "300 trees, leaf 10": RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=0),
                "100 trees, leaf 30, depth <= 12": RandomForestClassifier(100, min_samples_leaf=30, max_depth=12, n_jobs=-1,
                                                                         random_state=0)}.items():
    m.fit(X_tr, y_tr)
    m.set_params(n_jobs=1)                                                # online scoring: one row, one thread
    lat = []
    for i in range(30):
        t0 = time.perf_counter()
        m.predict_proba(X_te.iloc[[i]])
        lat.append((time.perf_counter() - t0) * 1000)
    rows[name] = {"total leaves": sum(e.get_n_leaves() for e in m.estimators_), "size MB": len(pickle.dumps(m)) / 1e6,
                  "latency ms (1 row)": np.median(lat), "test AUC": roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])}
dep = pd.DataFrame(rows).T
print(dep.to_string(float_format=lambda v: f"{v:,.3f}" if v < 100 else f"{v:,.0f}"))
note(f"""A default forest stores every leaf of every fully grown tree: here
{dep.iloc[0, 1]:.0f} MB for 8,400 training rows, and it grows with the data. Bigger leaves and fewer,
shallower trees shrink it {dep.iloc[0, 1] / dep.iloc[2, 1]:.0f}x with the SAME or better AUC. Model size drives memory,
container start-up time and cost: part of 'deployment complexity' in the
assignment report.""")

step("3.3  Forest probabilities vs the TRUE probabilities")
rows = {}
models = {"forest, leaf 1": RandomForestClassifier(300, n_jobs=-1, random_state=0),
          "forest, leaf 30": RandomForestClassifier(300, min_samples_leaf=30, n_jobs=-1, random_state=0),
          "forest, leaf 1 + isotonic calibration": CalibratedClassifierCV(RandomForestClassifier(300, n_jobs=-1, random_state=0),
                                                                          method="isotonic", cv=5)}
bands = pd.cut(pt_te, [0, 0.05, 0.2, 0.5, 1.0])
calib_rows = {}
for name, m in models.items():
    pr = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    rows[name] = {"test AUC": roc_auc_score(y_te, pr), "Brier": brier_score_loss(y_te, pr),
                  "mean |p - true p|": np.mean(np.abs(pr - pt_te))}
    calib_rows[name] = pd.Series(pr).groupby(np.asarray(bands.astype(str))).mean()
print(pd.DataFrame(rows).T.round(4).to_string())
calib = pd.DataFrame(calib_rows)
calib.insert(0, "TRUE mean prob", pd.Series(pt_te).groupby(np.asarray(bands.astype(str))).mean())
print("\n  Mean predicted probability, by band of TRUE risk:")
print(calib.round(3).to_string())
top_band = calib.index[-1]
note(f"""A forest averages trees, so its probabilities are pulled towards the middle:
for the truly high-risk band (true mean {calib.loc[top_band, 'TRUE mean prob']:.2f}) every version under-states the risk
({calib.loc[top_band].iloc[1:].min():.2f}-{calib.loc[top_band].iloc[1:].max():.2f}). Larger leaves give the closest numbers on average but squeeze the
top even more. Isotonic calibration helps only a little here: it learns from
the forest's OWN ranking, and with few high-risk applicants per fold the
correction is noisy. Calibration is not magic; check it per band, and if
probabilities feed a cost calculation (Module 4, Lab 06), measure the money
impact, not just the Brier score.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(n_tab.index, n_tab["OOB AUC"], "o-")
    axes[0].set(xscale="log", xlabel="n_estimators", ylabel="OOB AUC", title="More trees: never worse")
    k = knob.sort_values("OOB AUC")
    axes[1].barh(k.index, k["OOB AUC"], color="#4c78a8")
    axes[1].set(xlim=(k["OOB AUC"].min() - 0.01, k["OOB AUC"].max() + 0.005), xlabel="OOB AUC", title="The knobs")
    axes[1].tick_params(axis="y", labelsize=8)
    for name in models:
        axes[2].plot(calib["TRUE mean prob"], calib[name], "o-", label=name)
    axes[2].plot([0, 0.7], [0, 0.7], "k:", label="perfect")
    axes[2].set(xlabel="true probability (band mean)", ylabel="predicted (band mean)", title="Forest probabilities vs truth")
    axes[2].legend(fontsize=7)
    save_fig(fig, "04_random_forests")

banner("KEY TAKEAWAYS")
print("""  1. Random forest = deep trees on bootstrap samples + random feature subsets, averaged.
  2. Out-of-bag rows give a free validation score; it is fooled by duplicates and grouped rows.
  3. min_samples_leaf and max_features matter most; more trees never hurt, only cost time.
  4. Default forests are big: leaf size and depth cut size and latency with no loss of accuracy.
  5. Forest probabilities are squeezed towards the middle; calibrate when the numbers matter.""")
