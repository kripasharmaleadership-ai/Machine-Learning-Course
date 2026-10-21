"""
LAB 04 - Probabilities, Thresholds, the ROC Curve and the Precision-Recall Curve
================================================================================

FROM SCORE TO DECISION
    model -> score s(x) (usually a probability)  -> decision: positive if s(x) >= threshold t
    Lower t: more alerts -> recall UP, precision DOWN, FPR UP.  Higher t: the opposite.
    Every threshold gives one confusion matrix. A CURVE summarises all of them at once.

THE CURVES
    ROC curve : x = FPR = FP / (FP + TN),   y = TPR = recall      (one point per threshold)
                diagonal = random guessing; top-left corner = perfect
    ROC-AUC   : area under the ROC curve = P(a random positive scores higher than a random negative)
                0.5 = random, 1.0 = perfect ranking. Does NOT depend on the threshold or the base rate.
    PR curve  : x = recall, y = precision
                baseline = the prevalence (a random model's precision), not 0.5!
    Average precision (AP) : area under the PR curve; very sensitive to the base rate

ARE THE PROBABILITIES RIGHT? (calibration)
    A model is CALIBRATED if, among cases it gives 0.2, about 20% are positive.
    AUC only measures RANKING; calibration measures whether the NUMBERS can be trusted
    (needed for expected-cost decisions, Lab 06, and for any 'probability' shown to a human).

LEVELS
    EASY   : Scores by class, a threshold sweep: the precision-recall trade-off in a table.
    MEDIUM : ROC and PR curves by hand; AUC as a ranking probability; what AUC ignores.
    HARD   : Calibration (reliability diagrams, Brier, ECE) checked against the TRUE probabilities;
             fixing it; ROC-AUC vs AP when positives get rarer; the part of the curve that matters.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, NEG_COLOR, POS_COLOR, RANDOM_STATE, banner, fraud_features, make_transactions, note,
                          plt, save_fig, show, step)
from classification_toolkit import (auc_by_ranking, auc_trapezoid, average_precision, calibration_table,
                                    expected_calibration_error, pr_points, roc_points, sigmoid, threshold_sweep)

tx = make_transactions(n=40_000, seed=1, with_truth=True)
X, y = fraud_features(tx), tx["is_fraud"].to_numpy()
X_tr, X_te, y_tr, y_te, true_tr, true_te = train_test_split(X, y, tx["_true_prob"].to_numpy(), test_size=0.5,
                                                            stratify=y, random_state=RANDOM_STATE)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
p_lr = lr.predict_proba(X_te)[:, 1]

# %% =========================================================================
# LEVEL 1 (EASY): Scores, thresholds and the trade-off
# ============================================================================
banner("From a score to a decision", "EASY")
step("1.1  log-odds -> probability (the sigmoid)")
z = lr.decision_function(X_te.head(5))
print(pd.DataFrame({"log-odds z": z.round(2), "sigmoid(z) = 1/(1+e^-z)": sigmoid(z).round(4),
                    "predict_proba": p_lr[:5].round(4), "actual": y_te[:5]}).to_string())
note("""The model computes a score z (log-odds), and the sigmoid squashes it into
(0, 1). z = 0 -> 0.5; z = -4.6 -> 0.01; z = +2.2 -> 0.9. Everything below is
about what to DO with that number.""")

step("1.2  How the scores of the two classes overlap")
print(pd.DataFrame({"legit": pd.Series(p_lr[y_te == 0]).describe(percentiles=[.5, .9, .99]),
                    "fraud": pd.Series(p_lr[y_te == 1]).describe(percentiles=[.5, .9, .99])}).T.round(3).to_string())
note("""Most legit transactions get a tiny score, most frauds a higher one, but the two
distributions OVERLAP. Wherever we put the threshold, some frauds fall below
it (FN) and some legit transactions above it (FP). A threshold doesn't remove
errors; it chooses which kind you get.""")

step("1.3  The threshold sweep")
sweep = threshold_sweep(y_te, p_lr, thresholds=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
print(sweep[["threshold", "TP", "FP", "FN", "precision", "recall", "specificity", "alert rate"]].round(3).to_string(index=False))
note("""Lowering the threshold from 0.9 to 0.01 raises recall from a handful of frauds
to most of them, while the alerts grow from a few to thousands and precision
falls. There is no free lunch: the curve below is the menu, and the business
picks the dish (Lab 06).""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): ROC and PR curves by hand
# ============================================================================
banner("ROC and precision-recall curves, by hand", "MEDIUM")
fpr, tpr, thr = roc_points(y_te, p_lr)
fpr_sk, tpr_sk, _ = roc_curve(y_te, p_lr, drop_intermediate=False)
show("Thresholds on the curve (distinct scores)", len(thr) - 1)
show("ROC-AUC by trapezoids / sklearn", f"{auc_trapezoid(fpr, tpr):.4f} / {roc_auc_score(y_te, p_lr):.4f}")
idx = [np.argmin(np.abs(fpr - f)) for f in [0.001, 0.01, 0.05, 0.1, 0.3]]
print(pd.DataFrame({"threshold": thr[idx], "FPR": fpr[idx], "TPR (recall)": tpr[idx]}).round(4).to_string(index=False))
note("""Sort the test transactions by score, lower the threshold one score at a time,
and record (FPR, TPR) at each step: that IS the ROC curve. Its area, by the
trapezoid rule, matches scikit-learn to 4 decimals.""")

step("2.1  ROC-AUC is a ranking probability")
rng = np.random.default_rng(0)
pos, neg = p_lr[y_te == 1], p_lr[y_te == 0]
pairs = 200_000
wins = np.mean(rng.choice(pos, pairs) > rng.choice(neg, pairs))
show("AUC via ranks (Mann-Whitney)", f"{auc_by_ranking(y_te, p_lr):.4f}")
show(f"Share of {pairs:,} random (fraud, legit) pairs ranked right", f"{wins:.4f}")
note("""Pick a random fraud and a random legit transaction: AUC is the probability
that the fraud gets the higher score. That's why AUC is 'threshold-free': it
judges only the ORDER of the scores. 0.5 = coin flip, 1.0 = every fraud
above every legit transaction.""")

step("2.2  The precision-recall curve and its baseline")
prec, rec, _ = pr_points(y_te, p_lr)
show("Average precision by hand / sklearn", f"{average_precision(y_te, p_lr):.4f} / {average_precision_score(y_te, p_lr):.4f}")
show("PR baseline = prevalence", f"{y_te.mean():.4f}")
for r_target in [0.25, 0.5, 0.75, 0.9]:
    i = np.argmax(rec >= r_target)
    show(f"Precision when recall reaches {r_target:.0%}", f"{prec[i]:.3f}")
note("""The PR curve shows what the fraud team lives with: to catch 90% of frauds,
how many of our alerts are real? A random model's precision is the fraud rate
(~1.6%), so the PR baseline is a flat line near 0, not the 0.5 diagonal of
the ROC plot. AP summarises the curve; compare it to the prevalence, not to 0.5.""")

step("2.3  What AUC ignores")
squashed = p_lr ** 0.2
shifted = np.clip(p_lr + 0.3, 0, 1)
rows = {name: {"ROC-AUC": roc_auc_score(y_te, s), "mean predicted": s.mean(), "actual fraud rate": y_te.mean(),
               "log loss": log_loss(y_te, np.clip(s, 1e-6, 1 - 1e-6))}
        for name, s in {"model probabilities p": p_lr, "p ** 0.2": squashed, "p + 0.3": shifted}.items()}
print(pd.DataFrame(rows).T.round(4).to_string())
note("""Any transformation that keeps the ORDER leaves AUC unchanged. 'p + 0.3' claims
every transaction is at least 30% likely to be fraud, a useless probability,
and has exactly the same AUC. AUC answers 'does the model rank well?', not
'are the probabilities right?'. Log loss and calibration (Level 3) answer
the second question.""")

# %% =========================================================================
# LEVEL 3 (HARD): Calibration, and which curve to trust
# ============================================================================
banner("Can we trust the probabilities? And which curve should we report?", "HARD")
step("3.1  Three models, checked against the TRUE fraud probabilities")
models = {
    "Logistic regression": lr,
    "Gaussian naive Bayes": make_pipeline(StandardScaler(), GaussianNB()).fit(X_tr, y_tr),
    "Random forest": RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X_tr, y_tr),
}
probs = {name: m.predict_proba(X_te)[:, 1] for name, m in models.items()}
probs["Naive Bayes + isotonic calibration"] = CalibratedClassifierCV(
    make_pipeline(StandardScaler(), GaussianNB()), method="isotonic", cv=5).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
rows = {}
for name, p in probs.items():
    rows[name] = {"ROC-AUC": roc_auc_score(y_te, p), "AP": average_precision_score(y_te, p),
                  "Brier": brier_score_loss(y_te, p), "log loss": log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6)),
                  "ECE": expected_calibration_error(y_te, p), "mean |p - true p|": np.mean(np.abs(p - true_te))}
calib = pd.DataFrame(rows).T
print(calib.to_string(float_format=lambda v: f"{v:.4f}"))
print("\n  Reliability table, Gaussian naive Bayes (10 equal-size bins):")
nb_table = calibration_table(y_te, probs["Gaussian naive Bayes"])
print(nb_table.to_string(float_format=lambda v: f"{v:.4f}"))
note(f"""Because this data is synthetic, we KNOW each transaction's true fraud
probability, so we can measure directly how wrong each model's numbers are.
Logistic regression is close to the truth: it minimises log loss, which
rewards honest probabilities. Naive Bayes ranks reasonably but its numbers are
far too extreme (it multiplies evidence as if features were independent). The
reliability table shows it: its top bin promises {nb_table.mean_predicted.iloc[-1]:.0%} fraud; {nb_table.observed_rate.iloc[-1]:.0%} occurs.
Isotonic calibration (learned on held-out folds) fixes the numbers, Brier and
ECE drop, while the ranking barely changes.""")

step("3.2  ROC-AUC vs average precision when positives get rarer")
fresh = make_transactions(n=150_000, seed=7)
pf, yf = lr.predict_proba(fraud_features(fresh))[:, 1], fresh["is_fraud"].to_numpy()
legit_idx, fraud_idx = np.flatnonzero(yf == 0), np.flatnonzero(yf == 1)
rows = {}
for keep in [0.3, 0.1, 0.03, 0.01]:
    sub = np.r_[legit_idx, rng.choice(fraud_idx, max(int(len(fraud_idx) * keep / 0.3), 5), replace=False)]
    rows[f"fraud rate {yf[sub].mean():.2%}"] = {"ROC-AUC": roc_auc_score(yf[sub], pf[sub]),
                                               "average precision": average_precision_score(yf[sub], pf[sub]),
                                               "AP / prevalence (lift)": average_precision_score(yf[sub], pf[sub]) / yf[sub].mean()}
rare = pd.DataFrame(rows).T
print(rare.round(3).to_string())
note(f"""Same model, same patterns, fewer frauds. ROC-AUC stays ~{rare['ROC-AUC'].mean():.2f} (it is
computed within each class separately), while AP drops from {rare['average precision'].iloc[0]:.2f} to {rare['average precision'].iloc[-1]:.2f}, because
the same FPR now means a bigger PILE of false alarms per real fraud. For rare
events, ROC-AUC can look reassuring while the alert queue is mostly noise.
Report AP (and precision at the operating point) alongside ROC-AUC, and quote
the prevalence.""")

step("3.3  The part of the curve that matters")
rows = {}
for name, p in probs.items():
    f_, t_, _ = roc_curve(y_te, p)
    rows[name] = {"ROC-AUC": roc_auc_score(y_te, p), "recall at FPR = 0.5%": np.interp(0.005, f_, t_),
                  "recall at FPR = 2%": np.interp(0.02, f_, t_), "recall at FPR = 20%": np.interp(0.2, f_, t_)}
region = pd.DataFrame(rows).T
print(region.round(3).to_string())
auc_gap = region.loc["Logistic regression", "ROC-AUC"] - region.loc["Random forest", "ROC-AUC"]
rec_lr, rec_rf = region.loc["Logistic regression", "recall at FPR = 2%"], region.loc["Random forest", "recall at FPR = 2%"]
note(f"""A fraud team can only afford a low false-alarm rate (say FPR <= 2%: one good
customer in 50 declined). The ROC-AUC gap between logistic regression and the
forest is only {auc_gap:.3f}, which sounds negligible. At FPR = 2% it is recall {rec_lr:.0%}
vs {rec_rf:.0%}: {rec_lr / rec_rf - 1:.0%} more frauds stopped, every day. Meanwhile the area out at
FPR = 20% (where nobody will ever operate) counts just as much in the AUC as
this narrow region. Compare models WHERE you will operate: recall at a fixed
FPR, or precision at a fixed recall.""")

if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    bins = np.logspace(-4.5, 0, 50)
    for cls, color, label in [(0, NEG_COLOR, "legit"), (1, POS_COLOR, "fraud")]:
        v = np.clip(p_lr[y_te == cls], bins[0], 1)
        axes[0, 0].hist(v, bins=bins, color=color, alpha=0.6, label=label, weights=np.full(len(v), 1 / len(v)))
    axes[0, 0].axvline(0.1, color="k", ls="--", label="threshold 0.1")
    axes[0, 0].set(xscale="log", xlabel="predicted P(fraud) (log scale)", ylabel="share of the class",
                   title="Scores of the two classes overlap")
    axes[0, 0].legend()
    for name, p in probs.items():
        f_, t_, _ = roc_curve(y_te, p)
        axes[0, 1].plot(f_, t_, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
        pr_, rc_, _ = pr_points(y_te, p)
        axes[1, 0].plot(rc_, pr_, label=f"{name} (AP {average_precision_score(y_te, p):.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k:", label="random")
    axes[0, 1].axvspan(0, 0.02, color="gold", alpha=0.3, label="where the team operates")
    axes[0, 1].set(xlabel="FPR", ylabel="TPR (recall)", title="ROC curves")
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].axhline(y_te.mean(), color="k", ls=":", label=f"random = prevalence {y_te.mean():.3f}")
    axes[1, 0].set(xlabel="recall", ylabel="precision", title="Precision-recall curves")
    axes[1, 0].legend(fontsize=7)
    for name in ["Logistic regression", "Gaussian naive Bayes", "Naive Bayes + isotonic calibration"]:
        ct = calibration_table(y_te, probs[name], n_bins=10)
        keep = (ct.mean_predicted > 1e-4) & (ct.observed_rate > 0)              # log axes: drop empty / ~0 bins
        axes[1, 1].plot(ct.mean_predicted[keep], ct.observed_rate[keep], "o-", label=name)
    axes[1, 1].plot([1e-4, 1], [1e-4, 1], "k:", label="perfectly calibrated")
    axes[1, 1].set(xscale="log", yscale="log", xlim=(1e-4, 1.2), ylim=(1e-4, 1.2), xlabel="mean predicted probability",
                   ylabel="observed fraud rate", title="Reliability diagram (log-log, 10 equal-size bins)")
    axes[1, 1].legend(fontsize=7)
    save_fig(fig, "04_hard_roc_pr_calibration")

banner("KEY TAKEAWAYS")
print("""  1. A threshold turns a score into a decision; moving it trades recall for precision.
  2. ROC curve = (FPR, TPR) at every threshold; ROC-AUC = P(positive ranked above negative).
  3. PR curve: its baseline is the prevalence; AP falls as positives get rarer, ROC-AUC doesn't.
  4. AUC measures ranking only; calibration (Brier, ECE, reliability) checks the numbers.
  5. Compare models in the region you will operate in: recall at the FPR you can afford.""")
