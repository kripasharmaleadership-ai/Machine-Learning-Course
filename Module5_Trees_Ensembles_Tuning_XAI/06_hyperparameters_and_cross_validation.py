"""
LAB 06 - Hyperparameters vs Learned Parameters, and Cross-Validation
====================================================================

TWO KINDS OF NUMBERS IN A MODEL
    learned parameters   set by fit() from the training data       tree thresholds, leaf values, coefficients
    hyperparameters      set by YOU before fit()                   max_depth, learning_rate, C, n_estimators
    Choosing hyperparameters is itself learning from data -> it needs data the model was NOT trained on.

CROSS-VALIDATION (k-fold)
    split the training data into k folds; train on k-1, score on the held-out fold; repeat k times;
    average the k scores. Every row is used for validation exactly once.

    KFold            plain random folds
    StratifiedKFold  same class balance in every fold (classification default)
    GroupKFold       all rows of one group (customer, machine, patient) in the SAME fold
    TimeSeriesSplit  always train on the past, validate on the future

    The golden rule: the validation fold must look like FUTURE data. If rows are grouped or
    ordered in time, a random split lets the model peek at the answer.

LEVELS
    EASY   : Learned vs hyperparameters, counted for four models; the train / validation / test roles.
    MEDIUM : k-fold by hand; how noisy one validation score is; how many folds.
    HARD   : Three ways cross-validation lies: grouped rows, time drift, and preprocessing outside the folds.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_lab_utils import (CREDIT_NUM, HAS_MPL, MAINT_TARGET, RANDOM_STATE, banner, credit_true_logit,
                          make_credit_applicants, make_machine_telemetry, note, plt, save_fig, show, step)

credit = make_credit_applicants(n=8000, seed=6)
X = pd.get_dummies(credit[CREDIT_NUM + ["employment"]], columns=["employment"], dtype=float)
y = credit["default"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): What you set vs what the model learns
# ============================================================================
banner("Hyperparameters (you choose) vs learned parameters (fit chooses)", "EASY")
models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "decision tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0),
    "random forest": RandomForestClassifier(200, min_samples_leaf=10, n_jobs=-1, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, random_state=0),
}
rows = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    est = m[-1] if hasattr(m, "steps") else m
    if name == "logistic regression":
        learned = est.coef_.size + 1
        what = "1 weight per feature + intercept"
    elif name == "decision tree":
        learned = 2 * (est.tree_.node_count - est.get_n_leaves()) + est.get_n_leaves()
        what = "feature + threshold per split, value per leaf"
    elif name == "random forest":
        learned = sum(2 * (e.tree_.node_count - e.get_n_leaves()) + e.get_n_leaves() for e in est.estimators_)
        what = "every split and leaf of every tree"
    else:
        leaves = sum(len(pred[0].nodes) for pred in est._predictors)
        learned = leaves
        what = "every node of every boosted tree"
    rows[name] = {"hyperparameters you can set": len(est.get_params()), "learned numbers": learned, "what is learned": what,
                  "test AUC": roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}"))
print("""
    data split     used for                                 touched how often
    training       fit(): learn the PARAMETERS              every fit
    validation     choose the HYPERPARAMETERS (tune)        many times -> it gets 'used up'
    test           one final, honest estimate               ONCE
    (cross-validation = recycling the training data as many training/validation pairs)""")
note("""A tree learns a few hundred numbers; a forest hundreds of thousands. You set
~20 hyperparameters on every model, most of which you leave at their
defaults. The ones that matter (Labs 02-05) control the model's CAPACITY:
depth, leaf size, learning rate, regularisation. Picking them by looking at
test scores would turn the test set into a validation set, and you'd have no
honest estimate left.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Cross-validation mechanics
# ============================================================================
banner("k-fold cross-validation, by hand and by numbers", "MEDIUM")
step("2.1  5-fold CV by hand = cross_val_score")
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0)
folds = StratifiedKFold(5, shuffle=True, random_state=0)
by_hand = []
for k, (a, b) in enumerate(folds.split(X_tr, y_tr)):
    s = roc_auc_score(y_tr[b], tree.fit(X_tr.iloc[a], y_tr[a]).predict_proba(X_tr.iloc[b])[:, 1])
    by_hand.append(s)
    print(f"    fold {k + 1}: train on {len(a):,} rows, validate on {len(b):,} rows -> AUC {s:.3f}")
show("Mean of the 5 folds (by hand)", f"{np.mean(by_hand):.4f}  (sd {np.std(by_hand):.3f})")
show("cross_val_score", f"{cross_val_score(tree, X_tr, y_tr, cv=folds, scoring='roc_auc').mean():.4f}")

step("2.2  How noisy is ONE validation score?")
holdout, cv5 = [], []
for seed in range(40):
    a, b, ya, yb = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    holdout.append(roc_auc_score(yb, tree.fit(a, ya).predict_proba(b)[:, 1]))
for seed in range(10):
    cv5.append(cross_val_score(tree, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc").mean())
noise_tab = pd.DataFrame({"one 80/20 split (40 seeds)": pd.Series(holdout).describe()[["mean", "std", "min", "max"]],
                          "5-fold CV mean (10 seeds)": pd.Series(cv5).describe()[["mean", "std", "min", "max"]]}).T
print(noise_tab.round(4).to_string())
note(f"""The SAME model on the SAME data gets anywhere between {min(holdout):.3f} and {max(holdout):.3f} AUC
depending on which 20% happens to be held out. Averaging 5 folds shrinks that
spread by about {np.std(holdout) / np.std(cv5):.1f}x. Model differences smaller than this noise are not
differences (Module 3, Lab 08: bootstrap CIs).""")

step("2.3  How many folds?")
rows = {}
for k in [2, 3, 5, 10, 20]:
    t0 = time.perf_counter()
    sc = cross_val_score(tree, X_tr, y_tr, cv=StratifiedKFold(k, shuffle=True, random_state=0), scoring="roc_auc")
    rows[f"{k}-fold"] = {"training size per fit": int(len(y_tr) * (k - 1) / k), "mean AUC": sc.mean(),
                         "sd across folds": sc.std(), "seconds": time.perf_counter() - t0}
print(pd.DataFrame(rows).T.round(4).to_string())
note("""More folds = each model trains on more data (slightly less pessimistic) but
costs more fits, and each validation fold is smaller (noisier individually).
5 or 10 folds is the usual compromise; repeat with different shuffles
(RepeatedStratifiedKFold) when the data is small.""")

# %% =========================================================================
# LEVEL 3 (HARD): When cross-validation lies
# ============================================================================
banner("Three ways cross-validation lies", "HARD")

step("3.1  Grouped rows: the same machine in train and validation")
tele = make_machine_telemetry(n_machines=300, n_weeks=30, seed=5, robustness_sd=2.5)   # some pumps are 'repeat offenders'
F = ["install_age_years", "days_since_service", "temperature_c", "vibration_mm_s", "pressure_bar", "rpm", "load_factor"]
machines = tele.machine_id.unique()
rng = np.random.default_rng(0)
new_machines = rng.choice(machines, 75, replace=False)
seen = tele[~tele.machine_id.isin(new_machines)].reset_index(drop=True)
unseen = tele[tele.machine_id.isin(new_machines)]
rf = RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=0)
rows = {
    "random KFold (rows shuffled)": cross_val_score(rf, seen[F], seen[MAINT_TARGET], scoring="roc_auc",
                                                    cv=KFold(5, shuffle=True, random_state=0)).mean(),
    "GroupKFold by machine_id": cross_val_score(rf, seen[F], seen[MAINT_TARGET], scoring="roc_auc",
                                                cv=GroupKFold(5), groups=seen.machine_id).mean(),
    "reality: 75 machines never seen": roc_auc_score(unseen[MAINT_TARGET],
                                                     rf.fit(seen[F], seen[MAINT_TARGET]).predict_proba(unseen[F])[:, 1]),
}
grp = pd.Series(rows, name="ROC-AUC")
print(grp.round(3).to_string())
note(f"""Each pump appears in 30 weekly rows, and 'install_age_years' is unique per pump:
a fingerprint. With random folds, the forest recognises pumps it has already
seen in training and recalls how fragile each one is (a hidden property):
CV says {grp.iloc[0]:.3f}. Grouping the folds by machine says {grp.iloc[1]:.3f}, and new machines
really give {grp.iloc[2]:.3f}. Whenever rows share an entity (customer, patient, machine,
store), ask: will the model see THIS entity again in production? If not, use
GroupKFold.""")

step("3.2  A regime change: shuffled folds vs training on the past")
quarters = []
for q in range(8):
    d = make_credit_applicants(n=1500, seed=200 + q)
    logit = credit_true_logit(d)
    if q >= 6:                                       # a new card product changes who defaults
        logit = (logit - 1.7 * (d["debt_ratio"] > 0.45) - 1.4 * ((d["age"] < 30) & (d["utilisation"] > 0.7))
                 - 0.9 * (d["utilisation"] > 0.9) + 2.2 * (d["utilisation"] < 0.25))
    d["default"] = (np.random.default_rng(q).random(len(d)) < 1 / (1 + np.exp(-logit))).astype(int)
    d["quarter"] = q
    quarters.append(d)
drift = pd.concat(quarters, ignore_index=True)                       # sorted by quarter
Xd = pd.get_dummies(drift[CREDIT_NUM + ["employment"]], columns=["employment"], dtype=float)
yd = drift["default"].to_numpy()
past, future = (drift.quarter < 7).to_numpy(), (drift.quarter == 7).to_numpy()
hgb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0)
shuffled = cross_val_score(hgb, Xd[past], yd[past], scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0))
forward = cross_val_score(hgb, Xd[past], yd[past], scoring="roc_auc", cv=TimeSeriesSplit(6))   # 6 folds = quarters 1..6
reality = roc_auc_score(yd[future], hgb.fit(Xd[past], yd[past]).predict_proba(Xd[future])[:, 1])
show("Shuffled 5-fold CV on quarters 0-6 (mean)", f"{shuffled.mean():.3f}  folds {np.round(shuffled, 3).tolist()}")
print("  Forward-chaining folds (train on all earlier quarters, validate on the next one):")
for q, s_ in enumerate(forward, start=1):
    print(f"    validate on quarter {q}: AUC {s_:.3f}")
show("Reality: train on quarters 0-6, score quarter 7", f"{reality:.3f}")
ts = pd.Series({"shuffled 5-fold (mean)": shuffled.mean(), "forward folds (mean)": forward.mean(),
                "forward: latest fold": forward[-1], "reality: next quarter": reality})
note(f"""From quarter 6 a new card product changes who defaults (high utilisation stops
being risky, very LOW utilisation becomes risky). Shuffled folds mix the new
quarter into every fold and average it away: a reassuring {shuffled.mean():.3f}, with
every fold alike. Forward-chaining folds show the story in order: steady around
{forward[:-1].min():.2f}-{forward[:-1].max():.2f}, then the latest fold collapses to {forward[-1]:.3f}: an alarm. The model
trained on quarters 0-6 indeed scores only {reality:.3f} on quarter 7. If the world
can change, validate forward in time and LOOK AT THE LATEST FOLD, not just
the mean.""")

step("3.3  Preprocessing outside the folds: selecting features on ALL the data")
rng = np.random.default_rng(1)
Xn = pd.DataFrame(rng.normal(size=(300, 5000)))                       # 5,000 features of pure noise
yn = rng.integers(0, 2, 300)                                          # labels unrelated to anything
selector_all = SelectKBest(f_classif, k=20).fit(Xn, yn)               # WRONG: looks at every row's label
wrong = cross_val_score(LogisticRegression(max_iter=2000), selector_all.transform(Xn), yn, cv=5, scoring="roc_auc").mean()
right = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=2000)), Xn, yn,
                        cv=5, scoring="roc_auc").mean()
show("Select 20 of 5,000 noise features on ALL data, then CV", f"AUC {wrong:.3f}")
show("Selection INSIDE the pipeline (refit per fold)", f"AUC {right:.3f}")
note("""Labels are coin flips, so any honest estimate must be ~0.5. Choosing the 20
features most correlated with the labels BEFORE cross-validation lets every
validation fold's labels vote on which features are kept, and CV reports an
impressive AUC on pure noise. The same leak happens with scaling, imputation,
target encoding and resampling done before the split. Put every data-dependent
step inside a Pipeline so it is re-fitted on each training fold only.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].hist(holdout, bins=15, alpha=0.7, label="one 80/20 split")
    axes[0].hist(cv5, bins=8, alpha=0.7, label="5-fold CV mean")
    axes[0].set(xlabel="validation AUC (same model, same data)", title="How noisy is one validation score?")
    axes[0].legend()
    axes[1].plot(range(1, len(forward) + 1), forward, "o-", label="forward fold (validate on quarter k)")
    axes[1].axhline(shuffled.mean(), color="#e45756", ls="--", label="shuffled 5-fold mean")
    axes[1].scatter([7], [reality], color="k", zorder=3, label="reality: quarter 7")
    axes[1].set(xlabel="quarter", ylabel="ROC-AUC", title="A regime change: only forward folds see it")
    axes[1].legend(fontsize=8)
    axes[2].barh(["features chosen on all data", "selection inside the pipeline", "truth (pure noise)"], [wrong, right, 0.5],
                 color=["#e45756", "#4c78a8", "#9d9d9d"])
    axes[2].set(xlim=(0.3, 1), xlabel="cross-validated AUC", title="5,000 noise features, coin-flip labels")
    save_fig(fig, "06_cross_validation")

banner("KEY TAKEAWAYS")
print("""  1. Parameters are learned by fit(); hyperparameters are chosen by you, using validation data.
  2. One validation split is noisy; k-fold CV averages it out (5 or 10 folds, repeated if data is small).
  3. Rows that share an entity need GroupKFold; drifting data needs TimeSeriesSplit.
  4. Every data-dependent step (selection, scaling, encoding, resampling) goes inside the Pipeline.
  5. Validate the way you will deploy: new entities, the future, unseen preprocessing.""")
