"""
LAB 01 - Binary and Multiclass Classification
=============================================

WHAT IS CLASSIFICATION?
    Regression (Module 3) predicts a NUMBER: "this house is worth $312k".
    Classification predicts a CATEGORY:     "this loan will default: yes / no".

    Every classifier gives you TWO things:
        1. a SCORE / probability per class   predict_proba(X) -> [[0.83, 0.17], ...]
        2. a DECISION                        predict(X)       -> [0, ...]   (score >= threshold)
    The decision is a business choice layered on top of the score (Labs 04 and 06).

KINDS OF CLASSIFICATION PROBLEMS
    binary      : 2 classes                    default / repay, fraud / legit, churn / stay
    multiclass  : K > 2, exactly ONE per row   digit 0-9, support intent, product category
    multi-label : any NUMBER of labels per row "refund AND cancel" in one message, tags on a photo
    (ordinal    : classes with an order        credit rating AAA > AA > A - mistakes differ in size)

LEVELS
    EASY   : A binary classifier for loan defaults: classes, probabilities, decisions, baseline.
    MEDIUM : Multiclass with real handwritten digits: softmax by hand, one-vs-rest vs
             one-vs-one, the multiclass confusion matrix, multi-label vs multiclass.
    HARD   : The closed-world problem: a classifier MUST pick a known class, even for garbage.
             Confidence, the reject option, coverage vs accuracy, and top-k accuracy.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from ml_lab_utils import (HAS_MPL, LOAN_FEATURES, RANDOM_STATE, banner, make_loan_applicants, make_support_messages,
                          note, plot_confusion, plt, save_fig, show, step)

# %% =========================================================================
# LEVEL 1 (EASY): A binary classifier
# ============================================================================
banner("Binary classification: will this applicant default?", "EASY")
loans = make_loan_applicants(n=5000)
print(loans.head(6).to_string())
show("Classes and their counts", loans["default"].value_counts().to_dict())
show("Default rate (prevalence of class 1)", f"{loans['default'].mean():.1%}")
note("""The TARGET is a category: 1 = defaulted within 12 months, 0 = repaid. By
convention class 1 is the 'positive' class: the event we are looking for (default,
fraud, churn, disease). 'Positive' means 'present', not 'good'.""")

X, y = loans[LOAN_FEATURES], loans["default"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)

step("1.1  Two outputs: a probability and a decision")
proba = clf.predict_proba(X_te)
pred = clf.predict(X_te)
demo = X_te.head(6).assign(**{"P(repay)": proba[:6, 0].round(3), "P(default)": proba[:6, 1].round(3),
                              "decision (p >= 0.5)": pred[:6], "actual": y_te.head(6).to_numpy()})
print(demo.to_string())
show("Probabilities in each row sum to 1", bool(np.allclose(proba.sum(axis=1), 1)))
show("predict() == (P(default) >= 0.5)", bool(np.array_equal(pred, (proba[:, 1] >= 0.5).astype(int))))
note("""predict_proba gives one probability per class; predict() simply applies a
0.5 threshold to P(default). The threshold is NOT part of the model: the bank
can decide that 'P(default) >= 0.2 -> manual review'. Keep the probability; it
is far more useful than the 0/1 decision.""")

step("1.2  Always start with a baseline")
for strategy in ["most_frequent", "stratified"]:
    dummy = DummyClassifier(strategy=strategy, random_state=0).fit(X_tr, y_tr)
    show(f"Baseline '{strategy}' accuracy", f"{accuracy_score(y_te, dummy.predict(X_te)):.3f}")
show("Logistic regression accuracy", f"{accuracy_score(y_te, pred):.3f}")
caught = int(((pred == 1) & (y_te.to_numpy() == 1)).sum())
show("Defaulters caught by the model (of all)", f"{caught} of {int(y_te.sum())}")
note(f"""Predicting 'nobody defaults' is already {1 - y_te.mean():.0%} accurate, and useless:
it catches zero defaulters. The model's accuracy is only a few points higher,
and at the default 0.5 threshold it catches only {caught} of {int(y_te.sum())} defaulters.
Accuracy alone can't tell us if the model is useful. Labs 03-06 build the right tools.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Multiclass
# ============================================================================
banner("Multiclass: 10 handwritten digits (real data)", "MEDIUM")
digits = load_digits()
Xd, yd = digits.data, digits.target
show("Images x pixels (8x8, values 0-16)", Xd.shape)
show("Classes", np.unique(yd).tolist())
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.3, stratify=yd, random_state=RANDOM_STATE)

step("2.1  Softmax: one score per class -> probabilities that sum to 1")
multi = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Xd_tr, yd_tr)
z = multi.decision_function(Xd_te[:1])[0]                         # 10 raw scores (logits)
softmax = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
print(pd.DataFrame({"logit z_k": z.round(2), "exp(z_k)/sum": softmax.round(4),
                    "predict_proba": multi.predict_proba(Xd_te[:1])[0].round(4)}).T.to_string())
show("Predicted digit (argmax) / actual", f"{softmax.argmax()} / {yd_te[0]}")
note("""Multinomial logistic regression learns ONE linear score per class. Softmax
turns the 10 scores into 10 probabilities: exponentiate (all positive), then
divide by the sum (they add to 1). The prediction is the class with the highest
probability. This is EXACTLY how an LLM picks its next token, with ~100,000
classes (the vocabulary) instead of 10.""")

step("2.2  Three ways to handle K classes")
strategies = {
    "multinomial (softmax, 1 model)": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    "one-vs-rest (K binary models)": make_pipeline(StandardScaler(),
                                                   OneVsRestClassifier(LogisticRegression(C=0.1, max_iter=5000))),
    "one-vs-one (K(K-1)/2 models)": make_pipeline(StandardScaler(),
                                                  OneVsOneClassifier(LogisticRegression(C=0.1, max_iter=5000))),
}
rows = {}
for name, m in strategies.items():
    m.fit(Xd_tr, yd_tr)
    est = m[-1]
    n_models = len(est.estimators_) if hasattr(est, "estimators_") else 1
    rows[name] = {"binary models trained": n_models, "test accuracy": accuracy_score(yd_te, m.predict(Xd_te))}
print(pd.DataFrame(rows).T.to_string())
note("""Some algorithms are natively multiclass (softmax regression, trees, KNN,
forests). Others are built from BINARY classifiers: one-vs-rest trains 'is it a
3 or not?' for each class; one-vs-one trains a model for every pair (45 for 10
classes, 4,950 for 100). Accuracy is similar here; cost and calibration differ.""")

step("2.3  The multiclass confusion matrix: WHICH classes get mixed up?")
pred_d = strategies["multinomial (softmax, 1 model)"].predict(Xd_te)
cm = confusion_matrix(yd_te, pred_d)
print(pd.DataFrame(cm, index=[f"true {i}" for i in range(10)], columns=[f"p{i}" for i in range(10)]).to_string())
off = cm.copy()
np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
per_class = pd.Series(cm.diagonal() / cm.sum(axis=1), index=range(10))
show("Overall accuracy", f"{accuracy_score(yd_te, pred_d):.3f}")
show("Worst class (recall)", f"digit {per_class.idxmin()} ({per_class.min():.1%})")
show("Most common confusion", f"true {i} predicted as {j} ({off[i, j]} times)")
note("""Rows = actual class, columns = predicted. The diagonal is correct; every
off-diagonal cell is a specific mistake. A single accuracy number hides which
classes suffer. In a support bot, 'refund' mistaken for 'billing' is a mild
annoyance; 'unsafe' mistaken for 'billing' is an incident (Lab 12).""")

step("2.4  Multiclass vs multi-label")
msgs = make_support_messages(n=600, seed=3)
example = msgs[msgs.complex].iloc[0]
print(f"  message : {example.text!r}")
print(f"  multiclass label (exactly one) : {example.intent}")
labels_sets = [{r.intent} | ({"refund_request"} if "refund" in r.text else set()) | (
    {"cancel_subscription"} if "cancel" in r.text else set()) for r in msgs.itertuples()]
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels_sets)
show("Multi-label target shape (rows x labels)", Y.shape)
show("Messages with more than one label", f"{(Y.sum(axis=1) > 1).mean():.1%}")
note("""A multiclass model is forced to choose ONE intent per message, so the second
request is silently dropped. Multi-label = one yes/no classifier per label
(K binary problems; probabilities need NOT sum to 1). Choose the formulation
from the business need: 'route to one team' (multiclass) vs 'handle every
request in the message' (multi-label).""")

# %% =========================================================================
# LEVEL 3 (HARD): The closed-world problem
# ============================================================================
banner("A classifier always answers - even when it shouldn't", "HARD")
model = strategies["multinomial (softmax, 1 model)"]
rng = np.random.default_rng(0)
weird = {
    "real test digits": Xd_te,
    "random noise images": rng.uniform(0, 16, (300, 64)),
    "blank images (all 0)": np.zeros((50, 64)),
    "inverted digits (16 - x)": 16 - Xd_te[:300],
}
rows = {}
for name, data in weird.items():
    p = model.predict_proba(data)
    rows[name] = {"mean max-probability": p.max(axis=1).mean(), "share with max-prob > 0.9": (p.max(axis=1) > 0.9).mean(),
                  "most predicted class": int(np.bincount(p.argmax(axis=1), minlength=10).argmax())}
ood = pd.DataFrame(rows).T
ood["most predicted class"] = ood["most predicted class"].astype(int)
print(ood.to_string(float_format=lambda v: f"{v:.3f}"))
note(f"""Softmax probabilities ALWAYS sum to 1 over the KNOWN classes, so a closed-set
classifier has no way to say 'this is not a digit'. Plot twist: on random noise
it is MORE confident ({ood.iloc[1, 0]:.2f}) than on real digits ({ood.iloc[0, 0]:.2f}). Extreme
inputs push the linear scores far from zero, and softmax turns big score gaps
into near-certainty. The same thing happens when an intent classifier meets an
out-of-scope question, or a fraud model meets a new attack pattern. High
probability does not mean in-distribution.""")

step("3.1  The reject option: abstain when the top probability is low")
p_te = model.predict_proba(Xd_te)
conf, guess = p_te.max(axis=1), p_te.argmax(axis=1)
correct = guess == yd_te
rows = {}
for t in [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]:
    keep = conf >= t
    rows[f"answer if max-prob >= {t:.2f}"] = {"coverage (answered)": keep.mean(),
                                               "accuracy on answered": correct[keep].mean(),
                                               "errors per 1,000 inputs": (~correct & keep).mean() * 1000}
cov = pd.DataFrame(rows).T
print(cov.round(4).to_string())
noise_conf = model.predict_proba(weird["random noise images"]).max(axis=1)
show("Noise images rejected at threshold 0.9", f"{(noise_conf < 0.9).mean():.0%}")
note(f"""Selective classification: let the model answer only when confident and send
the rest to a human (or a bigger model). Coverage goes down, accuracy on the
answered cases goes up, and the error count drops sharply. But it does NOT
catch out-of-distribution inputs: only {(noise_conf < 0.9).mean():.0%} of the noise images are
rejected at 0.9, while {1 - cov.iloc[3, 0]:.0%} of REAL digits are. The confidence
filter removes the wrong things. That needs a separate 'is this in scope?'
check (Lab 12).""")

step("3.2  Top-k accuracy: is the right answer among the top 3?")
for k in [1, 2, 3]:
    show(f"Top-{k} accuracy", f"{top_k_accuracy_score(yd_te, p_te, k=k):.4f}")
note("""When a human or an agent picks from a short list (suggested replies, candidate
tools, product categories), top-k accuracy is the relevant metric: the right
answer is almost always in the top 2. Probabilities, not just argmax, are what
make this possible.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    plot_confusion(axes[0], cm, [str(d) for d in range(10)], title="Digits: confusion matrix")
    bins = np.linspace(0, 1, 26)
    axes[1].hist(conf, bins=bins, alpha=0.6, label="real test digits", density=True)
    axes[1].hist(noise_conf, bins=bins, alpha=0.6, label="random noise", density=True)
    axes[1].set(xlabel="max predicted probability", ylabel="density", title="Confidence is not 'in-distribution'")
    axes[1].legend()
    axes[2].plot(cov["coverage (answered)"], cov["accuracy on answered"], "o-")
    for lbl, r in cov.iterrows():
        axes[2].annotate(lbl.split(">= ")[1], (r["coverage (answered)"], r["accuracy on answered"]), fontsize=7,
                         xytext=(4, -10), textcoords="offset points")
    axes[2].set(xlabel="coverage (share answered)", ylabel="accuracy on answered", title="Reject option trade-off")
    save_fig(fig, "01_hard_closed_world")

banner("KEY TAKEAWAYS")
print("""  1. A classifier outputs a probability per class; the decision applies a threshold you choose.
  2. Always compare with a baseline: 'predict the majority class' can look very accurate.
  3. Multiclass = softmax over K scores (one model) or many binary models (OvR, OvO).
  4. Multi-label is a different problem: one yes/no decision per label.
  5. Closed-set models always answer; add a reject option and an out-of-scope check.""")
