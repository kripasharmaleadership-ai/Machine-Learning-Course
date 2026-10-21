"""
LAB 08 - K-Nearest Neighbours (KNN)
===================================

THE ALGORITHM (there is no training!)
    To classify a new point:
        1. measure its distance to EVERY training point   (usually Euclidean)
        2. take the k closest ones                        (its "neighbours")
        3. vote: P(class c) = share of the k neighbours in class c
    "Fit" just stores the data (a LAZY learner); all the work happens at prediction time.

THE KNOBS
    k            small k = local, wiggly, high variance;  large k = smooth, high bias
    scaling      distances add up features: a feature measured in thousands drowns one in 0..1
    weights      'uniform' (1 vote each) or 'distance' (closer neighbours count more)
    dimensions   in high dimensions every point is almost equally far away (the curse)

THE DATA (real, bundled with scikit-learn, works offline)
    Breast Cancer Wisconsin (Diagnostic): 569 tumour biopsies, 30 measurements of the cell
    nuclei in a digitised image (radius, texture, area, smoothness...). Positive class = malignant.

LEVELS
    EASY   : KNN by hand for one patient; scikit-learn agrees.
    MEDIUM : Choosing k (bias-variance); scaling is mandatory; probabilities come in steps.
    HARD   : The curse of dimensionality; prediction cost grows with the data; KNN as
             RETRIEVAL - explain a decision by its neighbours (the idea behind vector search).
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import HAS_MPL, LOAN_FEATURES, RANDOM_STATE, banner, make_loan_applicants, note, plt, save_fig, show, step

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, 1 - cancer.target.to_numpy()                 # sklearn codes benign = 1; we want malignant = 1
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
SHOW = ["mean radius", "mean texture", "mean area", "mean concave points", "worst smoothness"]


def cv_scores(model, Xa=X_tr, ya=y_tr):
    s = cross_validate(model, Xa, ya, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
    return {"CV accuracy": s["test_accuracy"].mean(), "CV recall (malignant)": s["test_recall"].mean(),
            "CV ROC-AUC": s["test_roc_auc"].mean()}


# %% =========================================================================
# LEVEL 1 (EASY): KNN by hand
# ============================================================================
banner("Diagnose one biopsy by asking its nearest neighbours", "EASY")
show("Biopsies / measurements", X.shape)
show("Malignant share", f"{y.mean():.1%}")
sc = StandardScaler().fit(X_tr)
Z_tr = sc.transform(X_tr)
new = X_te.iloc[[5]]
z_new = sc.transform(new)[0]
dist = np.sqrt(((Z_tr - z_new) ** 2).sum(axis=1))             # 1. distance to every training biopsy (30 features)
k = 7
nearest = np.argsort(dist)[:k]                                 # 2. the k closest
print("  New biopsy (5 of its 30 measurements):")
print(new[SHOW].to_string())
print(f"\n  Its {k} nearest training biopsies (distance over all 30 standardised features):")
print(X_tr.iloc[nearest][SHOW].assign(distance=dist[nearest].round(2), malignant=y_tr[nearest]).to_string())
p_hand = y_tr[nearest].mean()                                  # 3. vote
knn = KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y_tr)
show(f"P(malignant) by hand = malignant among {k}", f"{y_tr[nearest].sum()}/{k} = {p_hand:.3f}")
show("scikit-learn KNeighborsClassifier", f"{knn.predict_proba([z_new])[0, 1]:.3f}")
show("Actual diagnosis", "malignant" if y_te[5] else "benign")
note("""No equation, no coefficients: the model IS the training data. The prediction
is 'what the most similar past biopsies turned out to be'. Easy to explain,
but note what 'similar' means here: distance in standardised feature units.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): k, scaling and coarse probabilities
# ============================================================================
banner("The three things that make or break KNN", "MEDIUM")

step("2.1  Choosing k: from memorising to over-smoothing")
rows = {}
for kk in [1, 3, 5, 9, 15, 31, 75, 151]:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(kk))
    rows[kk] = {"train accuracy": m.fit(X_tr, y_tr).score(X_tr, y_tr), **cv_scores(m)}
ksweep = pd.DataFrame(rows).T
ksweep.index.name = "k"
print(ksweep.round(3).to_string())
best_k = int(ksweep["CV accuracy"].idxmax())
lr_scores = cv_scores(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)))
show("Best k by CV accuracy", best_k)
show("Logistic regression CV accuracy / recall", f"{lr_scores['CV accuracy']:.3f} / {lr_scores['CV recall (malignant)']:.3f}")
note(f"""k = 1 memorises: train accuracy 1.0, lower CV accuracy (high variance, Module 3,
Lab 04). Large k averages over so many neighbours that local detail is lost
(high bias): at k = 151 a quarter of the training set votes, and recall on
malignant tumours collapses, because the majority class (benign) wins every
close vote. CV picks k = {best_k}, roughly as accurate as logistic regression.""")

step("2.2  Scaling is mandatory")
rows = {"raw features": cv_scores(KNeighborsClassifier(best_k)),
        "standardised": cv_scores(make_pipeline(StandardScaler(), KNeighborsClassifier(best_k)))}
print(pd.DataFrame(rows).T.round(3).to_string())
print(X_tr[SHOW].std().round(3).rename("std of the raw feature").to_string())
note("""'mean area' has a standard deviation in the hundreds, 'worst smoothness' around
0.02. In raw units the distance is essentially 'difference in area', and the
other 29 measurements barely count. Standardising gives every feature the same
say. Always scale inside the pipeline.""")

step("2.3  KNN probabilities come in steps of 1/k")
p5 = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
show("Distinct probabilities, k = 5", sorted(np.unique(p5).round(2).tolist()))
show("Test biopsies with P = 0 or 1 exactly (k = 5)", f"{np.isin(p5, [0, 1]).mean():.0%}")
note("""With k = 5 the only possible 'probabilities' are 0, 0.2, 0.4, 0.6, 0.8, 1.
Most patients get exactly 0 or 1, which is not a credible statement of
certainty for a cancer diagnosis, and thresholds between the steps all give
the same decisions. Larger k (or weights='distance') gives finer scores;
calibration (Lab 04) can make them honest.""")

# %% =========================================================================
# LEVEL 3 (HARD): Dimensions, cost, and retrieval
# ============================================================================
banner("Where KNN breaks, and where it quietly powers modern AI", "HARD")

step("3.1  The curse of dimensionality: add irrelevant measurements")
rng = np.random.default_rng(0)
Z_te = sc.transform(X_te)
rows = {}
for n_junk in [0, 10, 30, 100, 300]:
    A_tr = np.c_[Z_tr, rng.normal(size=(len(Z_tr), n_junk))]
    A_te = np.c_[Z_te, rng.normal(size=(len(Z_te), n_junk))]
    kp = KNeighborsClassifier(best_k).fit(A_tr, y_tr).predict(A_te)
    lp = LogisticRegression(max_iter=5000).fit(A_tr, y_tr).predict(A_te)
    d = np.sqrt(((A_te[:, None, :] - A_tr[None, :, :]) ** 2).sum(-1))
    rows[f"30 real + {n_junk} junk"] = {"KNN accuracy": (kp == y_te).mean(), "KNN cancers missed": int(((kp == 0) & (y_te == 1)).sum()),
                                        "logistic accuracy": (lp == y_te).mean(),
                                        "logistic cancers missed": int(((lp == 0) & (y_te == 1)).sum()),
                                        "nearest / farthest distance": float(np.mean(d.min(1) / d.max(1)))}
curse = pd.DataFrame(rows).T.astype({"KNN cancers missed": int, "logistic cancers missed": int})
print(curse.to_string(float_format=lambda v: f"{v:.3f}"))
show("Malignant tumours in the test set", int(y_te.sum()))
note(f"""Every junk column adds random noise to every distance. With 300 junk features
the nearest biopsy is {curse.iloc[-1, -1]:.0%} as far away as the farthest one: 'nearest' no
longer means 'similar'. KNN's accuracy falls and the number of missed cancers
grows ({curse.iloc[0, 1]} -> {curse.iloc[-1, 1]}). Logistic regression learns to give the junk little weight
and suffers far less. KNN needs few, relevant, well-scaled features, or a
learned representation (embeddings) where distance means something.""")

step("3.2  No training cost, all the cost at prediction time")
big = make_loan_applicants(n=200_000, seed=4)
Xb, yb = StandardScaler().fit_transform(big[LOAN_FEATURES]), big["default"].to_numpy()
queries = Xb[:2000]
rows = {}
for n in [2_000, 20_000, 200_000]:
    t0 = time.perf_counter()
    kn = KNeighborsClassifier(15, algorithm="brute").fit(Xb[:n], yb[:n])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    kn.predict_proba(queries)
    pred_ms = (time.perf_counter() - t0) / len(queries) * 1000
    t0 = time.perf_counter()
    lrb = LogisticRegression(max_iter=2000).fit(Xb[:n], yb[:n])
    lr_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    lrb.predict_proba(queries)
    lr_ms = (time.perf_counter() - t0) / len(queries) * 1000
    rows[f"{n:,} training rows"] = {"KNN fit (s)": fit_s, "KNN ms / prediction": pred_ms,
                                    "logistic fit (s)": lr_fit, "logistic ms / prediction": lr_ms,
                                    "KNN stored data (MB)": Xb[:n].nbytes / 1e6}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))
note("""(Loan applicants, 5 features, to reach realistic sizes.) KNN 'fits' instantly
(it just stores the rows) but every prediction scans the stored data, so
latency and memory grow with the training set, while a logistic model is 6
numbers whatever the data size. Production KNN uses index structures
(KD-trees, ball trees) or APPROXIMATE nearest-neighbour indexes (HNSW, IVF):
exactly what vector databases are.""")

step("3.3  KNN is retrieval: explain a decision with its neighbours")
nn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(best_k)).fit(X_tr, y_tr)
p_all = nn_model.predict_proba(X_te)[:, 1]
case = int(np.argmin(np.abs(p_all - 0.5)))                      # the most uncertain test biopsy
dist_k, idx_k = nn_model[-1].kneighbors(nn_model[0].transform(X_te.iloc[[case]]), n_neighbors=best_k)
print("  The most uncertain test biopsy:")
print(X_te.iloc[[case]][SHOW].to_string())
print(f"\n  The {best_k} most similar past biopsies (the model's 'evidence'):")
print(X_tr.iloc[idx_k[0]][SHOW].assign(distance=dist_k[0].round(2), malignant=y_tr[idx_k[0]]).to_string())
show("P(malignant) / actual", f"{p_all[case]:.2f} / {'malignant' if y_te[case] else 'benign'}")
note("""A KNN decision comes with its own evidence: 'the most similar past biopsies
were split'. A pathologist can open those cases and judge for themselves. This
is case-based reasoning, and it is the core operation of modern AI retrieval:
embed a question as a vector, find the nearest stored vectors (documents,
past tickets, products), and use them. Same algorithm, learned features
(Module 6: embeddings and RAG).""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(ksweep.index, ksweep["train accuracy"], "o-", label="train accuracy")
    axes[0].plot(ksweep.index, ksweep["CV accuracy"], "o-", label="CV accuracy")
    axes[0].plot(ksweep.index, ksweep["CV recall (malignant)"], "o-", label="CV recall (malignant)")
    axes[0].axhline(lr_scores["CV accuracy"], color="grey", ls="--", label="logistic regression (CV acc)")
    axes[0].set(xscale="log", xlabel="k (log scale)", title="Choosing k")
    axes[0].legend(fontsize=8)
    n_junk = [0, 10, 30, 100, 300]
    axes[1].plot(n_junk, curse["KNN accuracy"], "o-", label="KNN")
    axes[1].plot(n_junk, curse["logistic accuracy"], "o-", label="logistic regression")
    axes[1].set(xlabel="irrelevant features added", ylabel="test accuracy", title="The curse of dimensionality")
    axes[1].legend()
    axes[2].hist(p5, bins=np.linspace(0, 1, 41), color="#4c78a8")
    axes[2].set(xlabel="predicted P(malignant), k = 5", ylabel="biopsies", title="KNN probabilities come in steps")
    save_fig(fig, "08_knn")

banner("KEY TAKEAWAYS")
print("""  1. KNN predicts by a vote of the k most similar training rows; there is no training step.
  2. Small k overfits, large k underfits (and the majority class wins): choose k by CV.
  3. Scale the features; distances are meaningless otherwise.
  4. Irrelevant features and high dimensions destroy 'nearest'; prediction cost grows with the data.
  5. KNN = retrieval: the same nearest-neighbour search powers vector databases and RAG.""")
