"""
LAB 03 - The Confusion Matrix, False Positives / False Negatives, and the Core Metrics
======================================================================================

THE CONFUSION MATRIX (binary, positive class = the event we look for: fraud, disease, churn)

                         predicted NEGATIVE     predicted POSITIVE
    actual NEGATIVE      TN  true negative      FP  false positive   ("false alarm", type I error)
    actual POSITIVE      FN  false negative     TP  true positive    ("miss", type II error)

THE METRICS (all built from those four numbers)
    accuracy    = (TP + TN) / all       share of decisions that are right
    precision   = TP / (TP + FP)        of the ALARMS we raised, how many were real?
    recall      = TP / (TP + FN)        of the REAL positives, how many did we catch?   (sensitivity, TPR)
    specificity = TN / (TN + FP)        of the real negatives, how many did we leave alone? (TNR)
    F1          = 2 * P * R / (P + R)   one number that is high only if BOTH P and R are high

    Memory aid: precision looks along the PREDICTED-positive column,
                recall and specificity look along the ACTUAL rows.

LEVELS
    EASY   : A confusion matrix by hand for 12 transactions; every metric by hand = scikit-learn;
             what an FP and an FN mean in five businesses.
    MEDIUM : A real fraud model: each metric answers a different stakeholder's question;
             why F1 uses the harmonic mean; F-beta when recall matters more.
    HARD   : The base-rate trap (precision depends on the population, recall doesn't);
             macro / micro / weighted averages hide a collapsed class; F1 depends on which
             class you call 'positive'; MCC and balanced accuracy.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, banner, fraud_features, make_transactions, note, plot_confusion, plt,
                          save_fig, show, step)
from classification_toolkit import binary_metrics, confusion_counts

# %% =========================================================================
# LEVEL 1 (EASY): A confusion matrix by hand
# ============================================================================
banner("Twelve transactions, four kinds of outcome", "EASY")
twelve = pd.DataFrame({
    "transaction": [f"T{i:02d}" for i in range(1, 13)],
    "actual": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    "predicted": [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
})
twelve["cell"] = np.select(
    [(twelve.actual == 1) & (twelve.predicted == 1), (twelve.actual == 0) & (twelve.predicted == 1),
     (twelve.actual == 1) & (twelve.predicted == 0)],
    ["TP (caught fraud)", "FP (false alarm)", "FN (missed fraud)"], "TN (left alone)")
print(twelve.to_string(index=False))
c = confusion_counts(twelve.actual, twelve.predicted)
print(f"""
                     predicted legit   predicted FRAUD
    actual legit        TN = {c['TN']}            FP = {c['FP']}
    actual FRAUD        FN = {c['FN']}            TP = {c['TP']}
""")
tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
step("Every metric by hand, then scikit-learn")
by_hand = {"accuracy (TP+TN)/all": (tp + tn) / 12, "precision TP/(TP+FP)": tp / (tp + fp),
           "recall TP/(TP+FN)": tp / (tp + fn), "specificity TN/(TN+FP)": tn / (tn + fp),
           "F1 2PR/(P+R)": 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / (tp / (tp + fp) + tp / (tp + fn))}
sk = [accuracy_score(twelve.actual, twelve.predicted), precision_score(twelve.actual, twelve.predicted),
      recall_score(twelve.actual, twelve.predicted),
      recall_score(twelve.actual, twelve.predicted, pos_label=0), f1_score(twelve.actual, twelve.predicted)]
for (name, v), s in zip(by_hand.items(), sk):
    show(name, f"{v:.3f}   (sklearn {s:.3f})")
note("""Read them in words. Accuracy: 9 of 12 decisions were right. Precision: of
the 4 transactions we BLOCKED, 3 were fraud. Recall: of the 4 frauds, we
caught 3. Specificity: of the 8 legit transactions, 6 went through. Note that
specificity is just 'recall of the negative class' (sklearn: pos_label=0).""")

step("1.1  What an FP and an FN cost: it depends entirely on the business")
fpfn = pd.DataFrame([
    ["Card fraud", "a good customer's card is declined (annoyed, may switch card)", "fraud goes through (bank loses the amount)"],
    ["Cancer screening", "a healthy patient gets a biopsy and weeks of anxiety", "a cancer is missed until it is advanced"],
    ["Spam filter", "a real email (an invoice, a job offer) lands in spam", "one spam email reaches the inbox"],
    ["Customer churn", "a retention discount is given to someone who'd have stayed", "a customer leaves without any offer"],
    ["Loan approval (1 = default)", "a good borrower is refused (lost interest income)", "a bad loan is granted (lost principal)"],
], columns=["problem", "FALSE POSITIVE means...", "FALSE NEGATIVE means..."])
print(fpfn.to_string(index=False))
note("""The confusion matrix is the same in every case; the PRICE of each cell is
not. A spam filter should hate FPs (lost invoices) far more than FNs. A cancer
screen should hate FNs far more than FPs. No metric is 'the best' until you
know these prices (Lab 06 turns them into money).""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): A real fraud model, read by different stakeholders
# ============================================================================
banner("One fraud model, four stakeholders, four metrics", "MEDIUM")
tx = make_transactions(n=40_000, seed=1)
X, y = fraud_features(tx), tx["is_fraud"].to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
p_te = model.predict_proba(X_te)[:, 1]
THRESHOLD = 0.15                                          # an operating point chosen by the fraud team
pred = (p_te >= THRESHOLD).astype(int)
m = binary_metrics(y_te, pred)
show("Test transactions / frauds", f"{len(y_te):,} / {y_te.sum()}")
show("Operating threshold", THRESHOLD)
print(pd.DataFrame([[m["TN"], m["FP"]], [m["FN"], m["TP"]]], index=["actual legit", "actual FRAUD"],
                   columns=["predicted legit", "predicted FRAUD"]).to_string())
stake = pd.DataFrame([
    ["Head of fraud", "Of the frauds, how many do we stop?", "recall", m["recall"]],
    ["Fraud analysts", "Of the alerts we review, how many are real?", "precision", m["precision"]],
    ["Customer experience", "Of the good customers, how many are left alone?", "specificity", m["specificity"]],
    ["Customer experience", "Good customers declined per 10,000", "FPR x 10,000", m["FPR"] * 10_000],
    ["The slide deck", "Share of all decisions that are right", "accuracy", m["accuracy"]],
], columns=["stakeholder", "question", "metric", "value"])
print(stake.to_string(index=False, float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:,.0f}"))
note(f"""Accuracy ({m['accuracy']:.1%}) is the least informative number in this table: it is
dominated by the {m['TN']:,} easy legit transactions. Each stakeholder needs a
different metric, and they move in opposite directions when the threshold
changes. A good model report shows ALL of them, plus the confusion matrix.""")

step("2.1  Why F1 is a HARMONIC mean")
everyone = np.ones_like(y_te)
mf = binary_metrics(y_te, everyone)
rows = {"our model": m, "flag EVERY transaction": mf}
tab = pd.DataFrame({k: {"precision": v["precision"], "recall": v["recall"],
                        "arithmetic mean": (v["precision"] + v["recall"]) / 2, "F1 (harmonic)": v["F1"]}
                    for k, v in rows.items()}).T
print(tab.round(3).to_string())
note("""Flagging everything gives recall 1.0 and precision = the fraud rate. Their
plain average is ~0.5, which looks respectable for a useless rule. The
harmonic mean 2PR / (P + R) is dragged down by the WEAKER of the two, so F1
is near zero. F1 is high only when precision AND recall are both high.""")

step("2.2  F-beta: when recall is worth more than precision")
grid = np.linspace(0.02, 0.6, 59)
best = {}
for beta in [0.5, 1, 2, 4]:
    scores = [fbeta_score(y_te, (p_te >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    t_best = grid[int(np.argmax(scores))]
    mb = binary_metrics(y_te, (p_te >= t_best).astype(int))
    best[f"F{beta} (recall {beta}x as important)"] = {"best threshold": t_best, "precision": mb["precision"],
                                                      "recall": mb["recall"], "alerts": mb["TP"] + mb["FP"]}
print(pd.DataFrame(best).T.round(3).to_string())
note("""F-beta weights recall beta times as much as precision. Maximising F2 or F4
chooses a LOWER threshold: more alerts, more frauds caught. F1 is just the
special case beta = 1, an implicit claim that a miss and a false alarm are
equally bad, which is rarely true. Better still: put real costs on them (Lab 06).""")
print("\n  The scikit-learn one-liner (per-class precision, recall, F1, support):")
print(classification_report(y_te, pred, target_names=["legit", "fraud"], digits=3))

# %% =========================================================================
# LEVEL 3 (HARD): Traps in reading the metrics
# ============================================================================
banner("The base-rate trap, averaging traps, and the 'positive' class", "HARD")

step("3.1  The base-rate trap: the same test, different populations")
sens, spec = 0.95, 0.95
rows = {}
for prev in [0.001, 0.01, 0.05, 0.2, 0.5]:
    n = 100_000
    pos = n * prev
    tp_, fn_ = sens * pos, (1 - sens) * pos
    tn_, fp_ = spec * (n - pos), (1 - spec) * (n - pos)
    rows[f"prevalence {prev:.1%}"] = {"recall": sens, "specificity": spec, "TP": tp_, "FP": fp_,
                                      "precision": tp_ / (tp_ + fp_), "accuracy": (tp_ + tn_) / n}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}" if v < 10 else f"{v:,.0f}"))
note("""A test with 95% sensitivity AND 95% specificity sounds excellent. Applied to a
population where 1 in 100 has the disease, only ~16% of positive results are
real: false alarms from the 99,000 healthy people swamp the 950 true cases.
Recall and specificity are properties of the TEST; precision depends on the
POPULATION it is applied to. This is the base-rate fallacy, and it fools
doctors, judges and data scientists alike.""")

step("3.1b  The same effect on our fraud model during a fraud wave")
fresh = make_transactions(n=120_000, seed=99)
pf = model.predict_proba(fraud_features(fresh))[:, 1]
legit_idx = np.flatnonzero(fresh["is_fraud"].to_numpy() == 0)
fraud_idx = np.flatnonzero(fresh["is_fraud"].to_numpy() == 1)
rng = np.random.default_rng(0)
rows = {}
for label, keep_share in [("normal month", 1.0), ("fraud wave (3x the fraud share)", 1 / 3.1),
                          ("big fraud wave (8x the fraud share)", 1 / 8.5)]:
    idx = np.r_[fraud_idx, rng.choice(legit_idx, int(len(legit_idx) * keep_share), replace=False)]
    mw = binary_metrics(fresh["is_fraud"].to_numpy()[idx], (pf[idx] >= THRESHOLD).astype(int))
    rows[label] = {"fraud rate": mw["prevalence"], "recall": mw["recall"], "specificity": mw["specificity"],
                   "precision": mw["precision"], "accuracy": mw["accuracy"]}
wave_tab = pd.DataFrame(rows).T
print(wave_tab.round(3).to_string())
note(f"""Same model, same threshold, same fraud PATTERNS, only a different mix of fraud
and legit traffic. Recall and specificity do not move; precision goes from
{wave_tab.precision.iloc[0]:.0%} to {wave_tab.precision.iloc[-1]:.0%}. (Accuracy drifts too.) When you report precision, always
report the prevalence it was measured at, and re-check it when the base rate
moves.""")

step("3.2  Macro, micro and weighted averages: one collapsed class")
digits = load_digits()
keep = (digits.target != 8) | (np.random.default_rng(0).random(len(digits.target)) < 0.08)
Xd, yd = digits.data[keep], digits.target[keep]
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.5, stratify=yd, random_state=RANDOM_STATE)
dm = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Xd_tr, yd_tr)
pd_te = dm.predict(Xd_te)
per = pd.DataFrame({"support": np.bincount(yd_te), "recall": recall_score(yd_te, pd_te, average=None),
                    "F1": f1_score(yd_te, pd_te, average=None)})
print(per.T.round(2).to_string())
avgs = {avg: f1_score(yd_te, pd_te, average=avg) for avg in ["micro", "weighted", "macro"]}
show("Accuracy", f"{accuracy_score(yd_te, pd_te):.3f}")
for k, v in avgs.items():
    show(f"F1 average='{k}'", f"{v:.3f}")
note(f"""Digit 8 is now rare (8% of its images kept); recall on it is {per.recall[8]:.0%}.
MICRO-F1 pools all decisions (= accuracy for single-label multiclass): it is
dominated by the common classes. WEIGHTED weights each class by its support,
so the rare class barely counts either. MACRO gives every class an equal
vote and is the only average that shows the collapse. If every class matters
(rare intents, rare diseases), report macro AND the per-class table.""")

step("3.3  F1 depends on which class you call 'positive'; MCC does not")
rows = {
    "fraud = positive": {"F1": f1_score(y_te, pred), "MCC": matthews_corrcoef(y_te, pred),
                         "balanced accuracy": balanced_accuracy_score(y_te, pred)},
    "legit = positive": {"F1": f1_score(1 - y_te, 1 - pred), "MCC": matthews_corrcoef(1 - y_te, 1 - pred),
                         "balanced accuracy": balanced_accuracy_score(1 - y_te, 1 - pred)},
}
print(pd.DataFrame(rows).T.round(3).to_string())
note(f"""The same predictions get F1 = {rows['fraud = positive']['F1']:.2f} or {rows['legit = positive']['F1']:.2f} depending on which class is
labelled 1, because F1 ignores the TN cell entirely. Matthews correlation
(MCC, -1 to 1, 0 = no better than chance) and balanced accuracy use all four
cells and are symmetric. Rule: make the RARE class the positive class, and
pair F1 with MCC or balanced accuracy when you need one number.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    plot_confusion(axes[0], confusion_matrix(y_te, pred), ["legit", "fraud"],
                   title=f"Fraud model at threshold {THRESHOLD}")
    prevs = np.logspace(-3.3, -0.3, 60)
    for s_, sp_ in [(0.95, 0.95), (0.9, 0.99), (0.99, 0.999)]:
        axes[1].plot(prevs, s_ * prevs / (s_ * prevs + (1 - sp_) * (1 - prevs)), label=f"recall {s_}, spec {sp_}")
    axes[1].set(xscale="log", xlabel="prevalence (log scale)", ylabel="precision", title="The base-rate trap")
    axes[1].legend(fontsize=8)
    cm_d = confusion_matrix(yd_te, pd_te)
    plot_confusion(axes[2], cm_d, [str(i) for i in range(10)], title=f"Rare digit 8: macro-F1 {avgs['macro']:.2f} "
                                                                     f"vs micro {avgs['micro']:.2f}")
    save_fig(fig, "03_hard_metric_traps")

banner("KEY TAKEAWAYS")
print("""  1. Confusion matrix: rows = actual, columns = predicted; FP = false alarm, FN = miss.
  2. Precision = quality of the alarms; recall = share of positives caught; specificity = negatives spared.
  3. F1 is the harmonic mean: high only if both precision and recall are high. F-beta tilts it.
  4. Precision depends on prevalence (base-rate trap); recall and specificity don't.
  5. Use macro averages for rare classes, make the rare class 'positive', add MCC.""")
