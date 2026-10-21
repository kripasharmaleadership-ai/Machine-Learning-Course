"""
LAB 08 - The Curse of Dimensionality
====================================

WHAT IT MEANS (Richard Bellman, 1961)
    As the number of features (dimensions) grows:
    1. SPACE EXPLODES   to keep 10 samples per axis you need 10^d samples
    2. DATA GETS SPARSE every point is far from every other point
    3. DISTANCES CONCENTRATE  nearest and farthest neighbour become almost equally
                        far -> "nearest" stops meaning anything (kNN, clustering, RAG search!)
    4. OVERFITTING      with more features than the data can support, a model finds
                        spurious patterns that look perfect on training data

WHERE IT BITES IN PRACTICE
    one-hot of high-cardinality columns, polynomial/interaction features, bag-of-words
    text, "let's join every table we have", genomics, embeddings with thousands of dims

REMEDIES
    feature selection (Lab 07) | domain features instead of raw dumps (Lab 06) |
    regularisation (L1/L2) | dimensionality reduction (PCA, embeddings) |
    models that are robust to irrelevant features (trees) | more data

LEVELS
    EASY   : Geometry - distance concentration, empty space, data requirements.
    MEDIUM : Add irrelevant features to a real problem and watch models degrade.
    HARD   : Feature explosion from polynomial features, and which remedies work
             (incl. why PCA can throw away the signal).
"""
# %%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ml_lab_utils import (HAS_MPL, LOAN_TARGET, RANDOM_STATE, banner, make_loan_applications, note, plt,
                          save_fig, show, step)
from preprocessing import LoanFeatureBuilder

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): The geometry of high dimensions
# ============================================================================
banner("High-dimensional space is weird", "EASY")

step("1.1  Distance concentration: nearest vs farthest neighbour")
dims = [1, 2, 5, 10, 50, 100, 500, 1000]
rows = []
for d in dims:
    pts = rng.uniform(size=(1000, d))
    q = rng.uniform(size=d)
    dist = np.linalg.norm(pts - q, axis=1)
    rows.append({"dims": d, "nearest": dist.min(), "farthest": dist.max(),
                 "relative contrast (far-near)/near": (dist.max() - dist.min()) / dist.min()})
conc = pd.DataFrame(rows)
print(conc.round(3).to_string(index=False))
note("""In 2-D the farthest point is hundreds of times farther than the nearest.
In 1,000-D, all 1,000 random points sit at almost the SAME distance. If
everything is equally far, 'nearest neighbour' is noise. This is why kNN,
k-means and vector search need GOOD, low-noise representations.""")

step("1.2  Empty space: most of a high-dimensional cube is 'corners'")
from math import gamma, pi  # noqa: E402

for d in [2, 3, 5, 10, 20]:
    ball = pi ** (d / 2) / gamma(d / 2 + 1) * 0.5 ** d          # inscribed ball, radius 0.5
    shell = 1 - 0.9 ** d                                        # volume within 5% of a face
    print(f"  d={d:>3}: inscribed ball holds {ball:7.2%} of the cube; outer 5% shell holds {shell:6.1%}")

step("1.3  Data needed to keep the same density (10 samples per axis)")
for d in [1, 2, 3, 6, 10, 20]:
    print(f"  d={d:>3}: {10.0 ** d:>26,.0f} samples")
note("""Real data does not fill the space - it lies on a much lower-dimensional
'manifold' (a credit applicant is not a random point in 50-D). Good features
and embeddings find that manifold; raw high-dimensional dumps do not.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Irrelevant features hurt - some models more than others
# ============================================================================
banner("Same problem, more and more irrelevant columns", "MEDIUM")
loans = LoanFeatureBuilder().fit_transform(make_loan_applications(n=4000, seed=81))
CORE = ["payment_to_income", "credit_score", "credit_utilization", "high_utilization",
        "delinquencies_2y", "age", "thin_file", "interest_rate"]
Xcore = loans[CORE].fillna(loans[CORE].median()).to_numpy()
y = loans[LOAN_TARGET].to_numpy()

MODELS = {
    "kNN (k=25)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
    "Logistic, weak penalty": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=100, max_iter=5000)),
    "Logistic, L2 (C=0.05)": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)),
    "Logistic, L1 (C=0.05)": lambda: make_pipeline(StandardScaler(), LogisticRegression(
        l1_ratio=1, C=0.05, solver="liblinear")),
    "Random forest": lambda: RandomForestClassifier(300, min_samples_leaf=5, max_features="sqrt",
                                                    n_jobs=-1, random_state=RANDOM_STATE),
}
extra_dims = [0, 20, 100, 500, 2000]
noise_all = rng.normal(size=(len(y), max(extra_dims)))
results = pd.DataFrame(index=list(MODELS), columns=[f"+{k} noise" for k in extra_dims], dtype=float)
gap = {}
for k in extra_dims:
    X = np.hstack([Xcore, noise_all[:, :k]])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    for name, mk in MODELS.items():
        m = mk().fit(Xtr, ytr)
        results.loc[name, f"+{k} noise"] = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        if name == "Logistic, weak penalty":
            gap[k] = roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])
print(results.round(3).to_string())
step("Train vs test AUC for the weakly-penalised logistic regression (2,400 training rows)")
print(pd.DataFrame({"train AUC": list(gap.values()), "test AUC": results.loc["Logistic, weak penalty"].to_numpy()},
                   index=[f"{8 + k} features" for k in extra_dims]).round(3).to_string())
note("""kNN collapses toward a coin flip: its distances are drowned by noise
dimensions (Level 1 in action). The weakly-penalised linear model OVERFITS:
with 2,000+ features for 2,400 rows it fits the training data almost perfectly
and generalises worse. L1 ignores most noise columns; random forests degrade
more slowly because each split picks the best of a random subset of features
(but with 2,000 noise columns, most candidate subsets contain no real signal).""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].semilogx(conc.dims, conc["relative contrast (far-near)/near"], marker="o")
    axes[0].set_xlabel("dimensions")
    axes[0].set_ylabel("(farthest - nearest) / nearest")
    axes[0].set_title("Distance concentration")
    for name in results.index:
        axes[1].plot([8 + k for k in extra_dims], results.loc[name], marker="o", label=name)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("total features (8 real + noise)")
    axes[1].set_ylabel("test AUC")
    axes[1].set_title("Adding irrelevant features")
    axes[1].legend(fontsize=7)
    save_fig(fig, "08_curse_of_dimensionality")

# %% =========================================================================
# LEVEL 3 (HARD): Feature explosion and remedies
# ============================================================================
banner("PolynomialFeatures explosion - and what actually fixes it", "HARD")
RAW = ["annual_income", "loan_amount", "interest_rate", "loan_term_months", "credit_score",
       "credit_utilization", "delinquencies_2y", "age", "num_credit_lines", "years_employed"]
loans_small = LoanFeatureBuilder().fit_transform(make_loan_applications(n=3000, seed=82))
Xr = loans_small[RAW].fillna(loans_small[RAW].median()).to_numpy()
yr = loans_small[LOAN_TARGET].to_numpy()
big = LoanFeatureBuilder().fit_transform(make_loan_applications(n=20000, seed=83))
Xbig = big[RAW].fillna(big[RAW].median()).to_numpy()
ybig = big[LOAN_TARGET].to_numpy()

step("3.1  'Let the model find interactions': polynomial degree 1 -> 3")
for deg in [1, 2, 3]:
    n_out = PolynomialFeatures(deg, include_bias=False).fit(Xr).n_output_features_
    print(f"  degree {deg}: {len(RAW)} raw features -> {n_out:,} features")

Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.4, stratify=yr, random_state=RANDOM_STATE)
rows = []
for deg in [1, 2, 3]:
    m = make_pipeline(StandardScaler(), PolynomialFeatures(deg, include_bias=False), StandardScaler(),
                      LogisticRegression(C=10, max_iter=20000)).fit(Xtr, ytr)
    rows.append({"degree": deg, "features": m[1].n_output_features_,
                 "train AUC": roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]),
                 "test AUC": roc_auc_score(yte, m.predict_proba(Xte)[:, 1])})
print(pd.DataFrame(rows).round(3).to_string(index=False))
note("""Degree 3 has every product of three raw columns - including the useful
loan x rate / income shape - buried among ~280 useless ones. With 1,800
training rows, the train/test gap widens: the model memorises.""")

step("3.2  Remedies, compared on the degree-3 explosion")
poly = lambda: [StandardScaler(), PolynomialFeatures(3, include_bias=False), StandardScaler()]     # noqa: E731
REMEDIES = {
    "Degree-3, weak penalty (baseline)": make_pipeline(*poly(), LogisticRegression(C=10, max_iter=20000)),
    "+ strong L2 regularisation": make_pipeline(*poly(), LogisticRegression(C=0.01, max_iter=20000)),
    "+ L1 regularisation": make_pipeline(*poly(), LogisticRegression(l1_ratio=1, C=0.05, solver="liblinear")),
    "+ SelectKBest(20), in pipeline": make_pipeline(*poly(), SelectKBest(f_classif, k=20),
                                                    LogisticRegression(max_iter=20000)),
    "+ PCA(20 components)": make_pipeline(*poly(), PCA(20, random_state=RANDOM_STATE),
                                          LogisticRegression(max_iter=20000)),
}
rows = []
for name, m in REMEDIES.items():
    m.fit(Xtr, ytr)
    rows.append({"approach": name, "test AUC": roc_auc_score(yte, m.predict_proba(Xte)[:, 1])})
domain = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
dom_cols = ["payment_to_income", "credit_score", "credit_utilization", "high_utilization", "delinquencies_2y",
            "age", "thin_file"]
Xd = loans_small[dom_cols].fillna(loans_small[dom_cols].median()).to_numpy()
Xd_tr, Xd_te = train_test_split(Xd, test_size=0.4, stratify=yr, random_state=RANDOM_STATE)
domain.fit(Xd_tr, ytr)
rows.append({"approach": "7 domain features, no polynomials", "test AUC": roc_auc_score(yte, domain.predict_proba(Xd_te)[:, 1])})
m_big = make_pipeline(*poly(), LogisticRegression(C=10, max_iter=20000)).fit(Xbig, ybig)
rows.append({"approach": "Degree-3 baseline, 20,000 training rows (more data)",
             "test AUC": roc_auc_score(yte, m_big.predict_proba(Xte)[:, 1])})
print(pd.DataFrame(rows).round(3).to_string(index=False))
note("""Regularisation and in-pipeline selection recover most of the loss. More
data helps too - the classic cure - but data is the most expensive remedy. The
cheapest one is domain knowledge: 7 well-chosen features match or beat 285
generated ones, with a model anyone can explain.""")

step("3.3  The PCA trap: maximum VARIANCE is not maximum SIGNAL")
note("""PCA is unsupervised: it keeps directions where the data varies most,
ignoring the target. If loud, irrelevant features dominate the variance, PCA
keeps them and throws away the quiet signal.""")
signal = StandardScaler().fit_transform(Xd)                         # 7 real features, std 1
loud_noise = rng.normal(0, 3, size=(len(yr), 30))                   # 30 sensor columns, std 3, no signal
Xp = np.hstack([signal, loud_noise])
Xp_tr, Xp_te = train_test_split(Xp, test_size=0.4, stratify=yr, random_state=RANDOM_STATE)
for name, m in [("All 37 features, logistic", make_pipeline(LogisticRegression(max_iter=5000))),
                ("PCA -> 7 components, logistic", make_pipeline(PCA(7), LogisticRegression(max_iter=5000))),
                ("SelectKBest(7) (supervised), logistic", make_pipeline(SelectKBest(f_classif, k=7),
                                                                        LogisticRegression(max_iter=5000)))]:
    m.fit(Xp_tr, ytr)
    show(name, roc_auc_score(yte, m.predict_proba(Xp_te)[:, 1]))
pca = PCA(7).fit(Xp_tr)
show("Variance 'explained' by the 7 PCs", f"{pca.explained_variance_ratio_.sum():.0%}")
show("Weight of the 7 REAL features in the PCs", f"{(pca.components_[:, :7] ** 2).sum() / 7:.1%}")
note("""PCA picked the 7 loudest directions - all noise - and kept almost none of
the real features. Note that standardising EVERYTHING first would hide this
trap here (then all columns tie on variance); in real data the 'loud' part is
often genuinely high-variance nuisance (seasonality, device effects).
Use PCA to compress correlated, comparable signals (pixels, sensor arrays,
embeddings) - and always check DOWNSTREAM performance, never just explained
variance. Learned embeddings (Lab 11) are trained to keep what matters for a
task, which is why they beat PCA on raw text counts.""")

banner("KEY TAKEAWAYS")
print("""  1. In high dimensions distances concentrate and data becomes sparse - nearest loses meaning.
  2. Irrelevant features hurt distance-based models most; weakly-regularised models overfit.
  3. Feature explosions (one-hot, polynomials, joins) need regularisation or selection.
  4. PCA maximises variance, not signal - validate it on the downstream task.
  5. The best dimensionality reduction is a few features that encode domain knowledge.""")
