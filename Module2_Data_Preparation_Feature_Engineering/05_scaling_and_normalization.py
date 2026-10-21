"""
LAB 05 - Scaling, Normalization and Log Transformation
======================================================

WHY SCALE?
    Income is in the tens of thousands, utilisation is between 0 and 1. Any
    algorithm that measures DISTANCE or uses GRADIENTS / PENALTIES treats "1 dollar"
    and "1.0 utilisation" as the same size step. Scaling puts features on a
    comparable footing.

THE TOOLS (all are column-wise, all must be FITTED ON TRAINING DATA ONLY)
    StandardScaler   z = (x - mean) / std          -> mean 0, std 1   (default choice)
    MinMaxScaler     (x - min) / (max - min)        -> [0, 1]          (bounded inputs, images)
    RobustScaler     (x - median) / IQR             -> outlier-resistant
    log1p            log(1 + x)                     -> tames right skew (money, counts)
    PowerTransformer Yeo-Johnson / Box-Cox          -> learns the best power transform
    QuantileTransformer                             -> forces a uniform/normal shape
    Normalizer       ROW-wise: x / ||x||            -> NOT a scaler! for text vectors / cosine

WHO CARES?
    Cares a lot : kNN, k-means, SVM, PCA, neural nets, regularised linear models
    Doesn't care: decision trees, random forests, gradient boosting (split thresholds)

LEVELS
    EASY   : Distances dominated by dollars; the three scalers side by side.
    MEDIUM : Which models care (measured); gradient descent speed; skew & log transforms.
    HARD   : Production bugs: per-batch normalisation, MinMax under drift, Normalizer
             confused with scaling, and the scaler as part of the model contract.
"""
# %%
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MinMaxScaler, Normalizer, PowerTransformer,
                                   QuantileTransformer, RobustScaler, StandardScaler)
from sklearn.svm import SVC

from ml_lab_utils import (HAS_MPL, LOAN_TARGET, RANDOM_STATE, banner, make_loan_applications, note,
                          plt, save_fig, show, step)

loans = make_loan_applications(n=8000, seed=51)
FEATS = ["annual_income", "loan_amount", "credit_score", "credit_utilization", "delinquencies_2y",
         "age", "num_credit_lines", "interest_rate"]
tr, te = train_test_split(loans, test_size=0.3, stratify=loans[LOAN_TARGET], random_state=RANDOM_STATE)
imp = SimpleImputer(strategy="median").fit(tr[FEATS])
Xtr = pd.DataFrame(imp.transform(tr[FEATS]), columns=FEATS)
Xte = pd.DataFrame(imp.transform(te[FEATS]), columns=FEATS)
y_tr, y_te = tr[LOAN_TARGET].to_numpy(), te[LOAN_TARGET].to_numpy()

# %% =========================================================================
# LEVEL 1 (EASY): Who is my nearest neighbour?
# ============================================================================
banner("Distances are dominated by the biggest units", "EASY")
cols = ["annual_income", "credit_score", "credit_utilization"]
query = Xtr[cols].iloc[[0]]
print("  Applicant:", query.round(3).to_dict("records")[0])
for name, X in [("unscaled", Xtr[cols]), ("standardized", StandardScaler().fit_transform(Xtr[cols]))]:
    X = np.asarray(X)
    nn = NearestNeighbors(n_neighbors=4).fit(X)
    idx = nn.kneighbors(X[[0]], return_distance=False)[0][1:]
    step(f"3 nearest neighbours - {name}")
    print(Xtr[cols].iloc[idx].round(3).to_string())
note("""Unscaled, 'nearest' just means 'similar income' - a credit score 150 points
apart counts like a $150 income difference. After standardising, each feature
contributes in units of its own spread, and the neighbours are similar on ALL
three features.""")

step("The three scalers on annual_income")
inc = Xtr[["annual_income"]]
summary = {}
for name, sc in [("raw", None), ("StandardScaler", StandardScaler()), ("MinMaxScaler", MinMaxScaler()),
                 ("RobustScaler", RobustScaler())]:
    v = inc.to_numpy().ravel() if sc is None else sc.fit_transform(inc).ravel()
    summary[name] = {"min": v.min(), "median": np.median(v), "mean": v.mean(), "max": v.max(), "std": v.std()}
print(pd.DataFrame(summary).T.round(3).to_string())

# %% =========================================================================
# LEVEL 2 (MEDIUM): Which models care? + log transforms
# ============================================================================
banner("Measured: which algorithms need scaling", "MEDIUM")
MODELS = {
    "kNN (k=25)": lambda: KNeighborsClassifier(25),
    "SVM (RBF)": lambda: SVC(),
    "Logistic (L2)": lambda: LogisticRegression(max_iter=5000),
    "Neural net": lambda: MLPClassifier((32,), max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
    "Random forest": lambda: RandomForestClassifier(200, min_samples_leaf=10, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}
SCALERS = {"none": None, "Standard": StandardScaler, "MinMax": MinMaxScaler, "Robust": RobustScaler}
sub = 3000                                     # SVM is slow; a subset is enough to see the effect
grid = pd.DataFrame(index=list(MODELS), columns=list(SCALERS), dtype=float)
for m_name, mk in MODELS.items():
    for s_name, sc in SCALERS.items():
        pipe = make_pipeline(sc(), mk()) if sc else make_pipeline(mk())
        pipe.fit(Xtr.iloc[:sub], y_tr[:sub])
        score = (pipe.decision_function(Xte) if hasattr(pipe[-1], "decision_function")
                 and not hasattr(pipe[-1], "predict_proba") else pipe.predict_proba(Xte)[:, 1])
        grid.loc[m_name, s_name] = roc_auc_score(y_te, score)
print(grid.round(3).to_string())
note("""kNN, SVM and the neural net jump once features are scaled - unscaled, they
effectively see only income. Trees give IDENTICAL results: a split 'income <
52,000' works the same as 'z_income < -0.31'. Logistic regression's AUC barely
moves here, but look at its training cost below, and remember its L2 penalty
is applied unevenly to features with different units. Scaling is never
harmful, often essential - and for tree models simply unnecessary.""")

step("2.1  Gradient-based training converges faster on scaled data")
for name, X in [("unscaled", Xtr), ("standardized", StandardScaler().fit_transform(Xtr))]:
    lr = LogisticRegression(max_iter=20000).fit(X, y_tr)
    show(f"lbfgs iterations to converge ({name})", int(lr.n_iter_[0]))

step("2.2  Skewed features and the log transform")
skewed = Xtr.annual_income.to_numpy()
transforms = {
    "raw": skewed,
    "log1p": np.log1p(skewed),
    "Yeo-Johnson (PowerTransformer)": PowerTransformer().fit_transform(skewed.reshape(-1, 1)).ravel(),
    "Quantile -> normal": QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)
    .fit_transform(skewed.reshape(-1, 1)).ravel(),
}
print(pd.Series({k: skew(v) for k, v in transforms.items()}, name="skewness").round(2).to_string())
note("""Skewness 0 = symmetric. Money is almost always right-skewed: many modest
values, a long tail of large ones. log1p is simple, explainable and invertible
('a 10% raise matters the same at every income'). PowerTransformer LEARNS the
best power; QuantileTransformer forces any shape into a bell curve (but squashes
real distances - outliers become 'merely large').""")

for name, fn in [("raw dollars", lambda X: X),
                 ("log1p(income, loan)", lambda X: X.assign(annual_income=np.log1p(X.annual_income),
                                                           loan_amount=np.log1p(X.loan_amount)))]:
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(fn(Xtr), y_tr)
    show(f"Logistic AUC with {name}", roc_auc_score(y_te, p.predict_proba(fn(Xte))[:, 1]))
note("""Why does log help a LINEAR model so much here? Default risk depends on
payment / income - a RATIO. log(loan) - log(income) = log(ratio), so on the log
scale a linear model can express the ratio by itself. Transformations are a
way of putting domain knowledge into the feature space (more in Lab 06).""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
    for ax, (name, v) in zip(axes, transforms.items()):
        ax.hist(v, bins=60, color="#4c78a8")
        ax.set_title(f"{name}\nskew={skew(v):.2f}", fontsize=9)
    save_fig(fig, "05_medium_skew_transforms")

# %% =========================================================================
# LEVEL 3 (HARD): Scaling bugs that reach production
# ============================================================================
banner("Four scaling bugs seen in real production systems", "HARD")
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, y_tr)

step("3.1  Bug: 'normalise each batch by itself' at serving time")
risky = Xte[(Xte.credit_score < 600) & (Xte.credit_utilization > 0.6)]
show("Batch: risky applicants only", len(risky))
right = model.predict_proba(risky)[:, 1].mean()
batch_scaler = StandardScaler().fit(risky)                      # the bug: statistics of THIS batch
wrong = model[-1].predict_proba(batch_scaler.transform(risky))[:, 1].mean()
show("Avg predicted default - training statistics", f"{right:.1%}")
show("Avg predicted default - per-batch statistics", f"{wrong:.1%}")
show("Actual default rate of this batch", f"{y_te[risky.index].mean():.1%}")
note("""Re-standardising the batch moves its mean to 0 - every risky applicant now
looks 'average'. Typical cause: a data engineer re-implements the scaling in
the serving code with df.mean() and df.std(). Fix: the scaler is FITTED ONCE on
training data and shipped INSIDE the pipeline artifact.""")

step("3.2  Bug: MinMax features under drift (a new 'jumbo' product raises loan sizes 60%)")
mm = MinMaxScaler().fit(Xtr[["loan_amount"]])
prod = Xte[["loan_amount"]] * 1.6
scaled = mm.transform(prod)
show("Training max loan_amount", f"${Xtr.loan_amount.max():,.0f}")
show("Max scaled value in production", float(scaled.max()))
show("Production rows outside [0, 1]", f"{((scaled > 1) | (scaled < 0)).mean():.1%}")
note("""MinMax promises [0, 1] only for the data it was fitted on. Downstream code
that ASSUMES [0,1] (sigmoid inputs, bucket edges, image pixels) silently breaks.
Monitor scaled-feature ranges - they are a cheap drift alarm.""")

step("3.3  Bug: Normalizer (row-wise) used instead of a scaler (column-wise)")
rows_normed = Normalizer().fit_transform(Xtr)
print(pd.DataFrame(rows_normed[:3], columns=FEATS).round(5).to_string())
p_norm = make_pipeline(Normalizer(), LogisticRegression(max_iter=5000)).fit(Xtr, y_tr)
show("Logistic AUC with Normalizer", roc_auc_score(y_te, p_norm.predict_proba(Xte)[:, 1]))
show("Logistic AUC with StandardScaler", roc_auc_score(y_te, model.predict_proba(Xte)[:, 1]))
note("""Normalizer divides each ROW by its length. With income in the row, the
length IS the income: income ~1.0, everything else ~0.0. It is the right tool
for TEXT vectors and EMBEDDINGS (cosine similarity, Lab 11), the wrong tool for
tabular features with different units.""")

step("3.4  The fitted scaler IS part of the model contract")
contract = pd.DataFrame({"train mean": model[0].mean_, "train std": model[0].scale_}, index=FEATS)
print(contract.round(3).to_string())
note("""These numbers are learned parameters, exactly like model weights. Version
them with the model, log them in the model card, and compare them with live
statistics to detect drift.""")

banner("KEY TAKEAWAYS")
print("""  1. Scale for distance/gradient/penalty-based models; trees do not need it.
  2. StandardScaler by default, RobustScaler with outliers, MinMax for bounded inputs.
  3. log1p for right-skewed money and counts - it can also turn ratios into sums.
  4. Fit on TRAIN only; never re-fit on a serving batch; ship the scaler inside the pipeline.
  5. Normalizer is row-wise (text vectors, embeddings), not a feature scaler.""")
