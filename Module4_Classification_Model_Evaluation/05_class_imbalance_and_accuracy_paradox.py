"""
LAB 05 - Class Imbalance, and Why Accuracy Can Be Misleading
============================================================

THE ACCURACY PARADOX
    If 98.4% of transactions are legit, the rule "nothing is ever fraud" is 98.4% accurate
    and catches zero frauds. With rare positives, accuracy measures how rare the positives
    are, not how good the model is.

THE TOOLBOX FOR IMBALANCED DATA
    1. Evaluate properly    stratified splits, recall / precision / AP, never accuracy alone
    2. Move the threshold   keep the model, lower the cut-off (Lab 04)
    3. Re-weight            class_weight="balanced": each fraud counts ~30x in the loss
    4. Re-sample            undersample the majority, oversample the minority, SMOTE
                            ONLY inside the training folds - never before the split
    5. Get more positives   often the real fix: imbalance hurts because there are FEW positives

LEVELS
    EASY   : The accuracy paradox; stratified vs plain splits.
    MEDIUM : Six imbalance strategies, evaluated identically. Surprise: they mostly move the
             threshold, and the re-weighting / re-sampling ones break the probabilities.
    HARD   : Resampling before the split (leakage that inflates scores); repairing probabilities
             with the prior-shift formula; few positives vs low ratio.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, banner, fraud_features, make_transactions, note, plt, save_fig, show,
                          step)
from classification_toolkit import binary_metrics, prior_shift, smote

tx = make_transactions(n=40_000, seed=1)
X, y = fraud_features(tx).to_numpy(), tx["is_fraud"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): The accuracy paradox
# ============================================================================
banner("98% accurate and completely useless", "EASY")
show("Fraud rate", f"{y.mean():.2%}  ({y.sum()} frauds in {len(y):,} transactions)")
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
p = lr.predict_proba(X_te)[:, 1]
cands = {
    "A: 'never fraud' (majority class)": DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te),
    "B: logistic regression, threshold 0.5": (p >= 0.5).astype(int),
    "C: logistic regression, threshold 0.05": (p >= 0.05).astype(int),
}
rows = {k: {m: binary_metrics(y_te, v)[m] for m in ["accuracy", "recall", "precision", "TP", "FP", "balanced accuracy"]}
        for k, v in cands.items()}
acc_tab = pd.DataFrame(rows).T.astype({"TP": int, "FP": int})
print(acc_tab.to_string(float_format=lambda v: f"{v:.3f}"))
note(f"""Ranked by accuracy, A ({acc_tab.iloc[0, 0]:.1%}) is nearly as good as B and BETTER than C. Ranked by
what the bank cares about, A stops 0 frauds, B stops {int(acc_tab.iloc[1]['TP'])}, C stops {int(acc_tab.iloc[2]['TP'])}. Accuracy
rewards the {(y_te == 0).sum():,} easy 'legit' decisions and barely notices the {y_te.sum()} that matter.
Balanced accuracy (the average of recall and specificity) ranks them sensibly.""")

step("1.1  Stratified splits keep the rare class in every fold")
small = np.arange(2000)                                    # a small project: 2,000 transactions
rows = {}
for name, cv in {"KFold (shuffled)": KFold(5, shuffle=True, random_state=3),
                 "StratifiedKFold": StratifiedKFold(5, shuffle=True, random_state=3)}.items():
    counts = [int(y[small][va].sum()) for _, va in cv.split(X[small], y[small])]
    rows[name] = {"frauds per validation fold": counts, "min": min(counts), "max": max(counts)}
print(pd.DataFrame(rows).T.to_string())
note("""With only ~30 frauds, a plain random split can leave a fold with very few
of them, and that fold's recall is computed on a handful of cases. Stratified
splitting keeps the fraud rate equal in every fold. Use stratify= in
train_test_split and StratifiedKFold for every classification problem.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Six strategies, evaluated identically
# ============================================================================
banner("Re-weighting, re-sampling, SMOTE, or just moving the threshold?", "MEDIUM")


def undersample(Xa, ya, ratio=1.0, seed=0):
    """Keep every fraud and a random subset of legit transactions (ratio = legit per fraud)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(ya == 1), np.flatnonzero(ya == 0)
    keep = np.r_[pos, rng.choice(neg, int(len(pos) * ratio), replace=False)]
    return Xa[keep], ya[keep]


def oversample(Xa, ya, seed=0):
    """Duplicate random frauds until both classes have the same size."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(ya == 1)
    extra = rng.choice(pos, (ya == 0).sum() - len(pos), replace=True)
    return np.vstack([Xa, Xa[extra]]), np.r_[ya, ya[extra]]


def scaled_smote(Xa, ya, seed=0):
    sc = StandardScaler().fit(Xa)                  # SMOTE needs distances -> scale first, then undo
    Xs, ys = smote(sc.transform(Xa), ya, k=5, seed=seed)
    return sc.inverse_transform(Xs), ys


base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
STRATEGIES = {
    "1 nothing (threshold 0.5)": (base, None),
    "2 class_weight='balanced'": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,
                                                                                   class_weight="balanced")), None),
    "3 random undersampling 1:1": (base, undersample),
    "4 random oversampling 1:1": (base, oversample),
    "5 SMOTE 1:1": (base, scaled_smote),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
oof = {}
for name, (model, resample) in STRATEGIES.items():
    o = np.zeros(len(y_tr))
    for k, (a, b) in enumerate(cv.split(X_tr, y_tr)):
        Xa, ya = (X_tr[a], y_tr[a]) if resample is None else resample(X_tr[a], y_tr[a], seed=k)  # inside the fold!
        o[b] = clone(model).fit(Xa, ya).predict_proba(X_tr[b])[:, 1]
    oof[name] = o
# strategy 6: no re-balancing, but pick the threshold on the out-of-fold predictions
# that gives the same recall as class_weight='balanced' at 0.5
target_recall = binary_metrics(y_tr, oof["2 class_weight='balanced'"] >= 0.5)["recall"]
t6 = np.quantile(oof["1 nothing (threshold 0.5)"][y_tr == 1], 1 - target_recall)
oof["6 nothing + threshold moved"] = oof["1 nothing (threshold 0.5)"]
thresholds = {k: 0.5 for k in oof} | {"6 nothing + threshold moved": t6}
rows = {}
for name, o in oof.items():
    m = binary_metrics(y_tr, o >= thresholds[name])
    rows[name] = {"threshold": thresholds[name], "ROC-AUC": roc_auc_score(y_tr, o), "AP": average_precision_score(y_tr, o),
                  "recall": m["recall"], "precision": m["precision"], "alerts": m["TP"] + m["FP"],
                  "mean predicted P": o.mean(), "Brier": brier_score_loss(y_tr, o)}
strat = pd.DataFrame(rows).T
print(strat.to_string(float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:,.0f}"))
show("Actual fraud rate", f"{y_tr.mean():.3f}")
note(f"""Read the table column by column (5-fold out-of-fold predictions):
- RANKING (ROC-AUC, AP) is almost the same for every strategy: re-weighting and
  re-sampling did not make the model better at telling fraud from legit.
- RECALL at 0.5 jumps for strategies 2-5, because they all shift the scores up:
  at 0.5 they behave like the plain model with a LOWER threshold. Strategy 6
  does exactly that, explicitly, and lands in the same place.
- PROBABILITIES are broken by 2-5: the average predicted fraud probability is
  ~{strat['mean predicted P'].iloc[1]:.2f} for a {y_tr.mean():.3f} fraud rate (look at Brier). Any expected-cost
  calculation (Lab 06) built on them would be wrong.
For a well-specified model like this one, 'fix imbalance' mostly means 'choose
the threshold'. Re-weighting helps more for models that struggle to learn
the rare class at all (deep trees, small data), and SMOTE is no magic.""", keep_lines=True)

# %% =========================================================================
# LEVEL 3 (HARD): Leakage, repairs, and what imbalance really costs
# ============================================================================
banner("Resampling traps and the real problem", "HARD")

step("3.1  Oversampling BEFORE the split: duplicated frauds in train AND validation")
Xs_, ys_ = X[:12_000], y[:12_000]
X_dev, X_hold, y_dev, y_hold = train_test_split(Xs_, ys_, test_size=0.3, stratify=ys_, random_state=1)
forest = RandomForestClassifier(200, n_jobs=-1, random_state=0)
rows = {}
Xo, yo = oversample(X_dev, y_dev)                            # WRONG: before the CV split
o_wrong = np.zeros(len(yo))
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xo, yo):
    o_wrong[b] = clone(forest).fit(Xo[a], yo[a]).predict_proba(Xo[b])[:, 1]
o_right = np.zeros(len(y_dev))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X_dev, y_dev)):
    Xa, ya = oversample(X_dev[a], y_dev[a], seed=k)          # RIGHT: inside each training fold
    o_right[b] = clone(forest).fit(Xa, ya).predict_proba(X_dev[b])[:, 1]
final = clone(forest).fit(*oversample(X_dev, y_dev))
p_hold = final.predict_proba(X_hold)[:, 1]
leak_tab = pd.DataFrame({
    "oversample, THEN cross-validate (wrong)": {"CV AP": average_precision_score(yo, o_wrong),
                                                "CV recall @0.5": binary_metrics(yo, o_wrong >= .5)["recall"]},
    "cross-validate, oversample inside each fold": {"CV AP": average_precision_score(y_dev, o_right),
                                                    "CV recall @0.5": binary_metrics(y_dev, o_right >= .5)["recall"]},
    "reality: untouched hold-out set": {"CV AP": average_precision_score(y_hold, p_hold),
                                        "CV recall @0.5": binary_metrics(y_hold, p_hold >= .5)["recall"]},
}).T
print(leak_tab.round(3).to_string())
note(f"""Oversampling copies each fraud ~{(y_dev == 0).sum() // y_dev.sum()} times. Done before the split, copies of the
SAME fraud land in training and validation folds; a random forest memorises
them, and the validation 'finds' frauds it has literally seen. CV AP {leak_tab.iloc[0, 0]:.2f}
vs a real {leak_tab.iloc[2, 0]:.2f}. The same leak happens with SMOTE (synthetic points are
interpolated from validation frauds). Rule: every step that LEARNS from data,
including resampling, goes inside the cross-validation loop.""")

step("3.2  Repairing probabilities after re-sampling (prior-shift correction)")
p_under = clone(base).fit(*undersample(X_tr, y_tr)).predict_proba(X_te)[:, 1]
p_fixed = prior_shift(p_under, train_rate=0.5, new_rate=y_tr.mean())
p_plain = lr.predict_proba(X_te)[:, 1]
rows = {name: {"ROC-AUC": roc_auc_score(y_te, q), "mean predicted": q.mean(), "Brier": brier_score_loss(y_te, q)}
        for name, q in {"plain model": p_plain, "trained on 1:1 undersample": p_under,
                        "1:1 undersample + prior-shift fix": p_fixed}.items()}
print(pd.DataFrame(rows).T.round(4).to_string())
note("""A model trained on a 50/50 sample believes frauds are 50% of the world. Bayes'
rule converts its odds back: odds_true = odds_model x (true odds of fraud) /
(training odds of fraud). The ranking is untouched (same AUC) and the average
probability is back at the real fraud rate. Undersampling + this correction is a
legitimate way to train on huge datasets cheaply.""")

step("3.3  Is it the RATIO or the NUMBER of positives?")
big = make_transactions(n=400_000, seed=5)
Xb, yb = fraud_features(big).to_numpy(), big["is_fraud"].to_numpy()
pos_all, neg_all = np.flatnonzero(yb == 1), np.flatnonzero(yb == 0)
rng = np.random.default_rng(0)
test_idx = np.r_[pos_all[:1500], neg_all[:90_000]]
pos_pool, neg_pool = pos_all[1500:], neg_all[90_000:]
forest_small = RandomForestClassifier(150, min_samples_leaf=3, n_jobs=-1, random_state=0)
grid = {}
for n_pos in [25, 100, 400, 1600]:
    for ratio in [10, 50, 150]:
        idx = np.r_[rng.choice(pos_pool, n_pos, replace=False), rng.choice(neg_pool, n_pos * ratio, replace=False)]
        pt = clone(forest_small).fit(Xb[idx], yb[idx]).predict_proba(Xb[test_idx])[:, 1]
        grid.setdefault(f"{n_pos} frauds", {})[f"1:{ratio}"] = average_precision_score(yb[test_idx], pt)
grid_tab = pd.DataFrame(grid).T
print("  Test average precision (random forest), rows = frauds in training, columns = fraud:legit ratio")
print(grid_tab.round(3).to_string())
note(f"""Read DOWN a column: more frauds = a much better model (AP {grid_tab.iloc[0, 1]:.2f} -> {grid_tab.iloc[-1, 1]:.2f} at 1:50).
Read ACROSS a row: adding more legit transactions (a MORE imbalanced training
set) does not hurt; it usually helps, because the model sees more of what
'normal' looks like. Imbalance is not the disease; too FEW positives is. The best
investments: more labelled positives (longer history, better labelling, data
sharing), then evaluation that respects the rarity. Throwing data away to
'balance' the classes rarely pays.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    names = list(strat.index)
    axes[0].barh(names, strat["recall"], color="#4c78a8", label="recall")
    axes[0].barh(names, strat["AP"], color="#f58518", alpha=0.7, label="average precision")
    axes[0].set(title="Recall changes, ranking (AP) doesn't")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=7)
    bins = np.linspace(0, 1, 41)
    axes[1].hist(oof["1 nothing (threshold 0.5)"], bins=bins, alpha=0.6, label="plain model", density=True)
    axes[1].hist(oof["2 class_weight='balanced'"], bins=bins, alpha=0.6, label="class_weight='balanced'", density=True)
    axes[1].set(yscale="log", xlabel="predicted P(fraud)", title="Re-weighting inflates every probability")
    axes[1].legend(fontsize=8)
    for col in grid_tab.columns:
        axes[2].plot([int(i.split()[0]) for i in grid_tab.index], grid_tab[col], "o-", label=f"fraud:legit {col}")
    axes[2].set(xscale="log", xlabel="frauds in training (log)", ylabel="test AP", title="More positives > better ratio")
    axes[2].legend(fontsize=8)
    save_fig(fig, "05_hard_imbalance")

banner("KEY TAKEAWAYS")
print("""  1. With rare positives, accuracy rewards doing nothing: report recall, precision, AP.
  2. Stratify every split so every fold contains the rare class.
  3. Re-weighting and re-sampling mostly move the effective threshold and break the probabilities.
  4. Resample only inside the training folds; fix probabilities with the prior-shift formula.
  5. The real enemy is too FEW positives, not the ratio: collect more labelled positives.""")
