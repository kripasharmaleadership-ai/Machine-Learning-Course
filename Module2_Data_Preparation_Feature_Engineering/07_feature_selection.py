"""
LAB 07 - Feature Selection: Correlation Analysis and Mutual Information
=======================================================================

WHY SELECT FEATURES?
    fewer features = less overfitting, faster training & serving, cheaper data
    (every bureau call costs money), simpler monitoring, easier explanations.

THREE FAMILIES
    FILTER    score each feature on its own, before modelling (fast, model-agnostic)
              - variance threshold      constant / near-constant columns
              - correlation (Pearson)   LINEAR relationships; also finds REDUNDANT pairs
              - Spearman                MONOTONIC relationships (rank-based)
              - mutual information (MI) ANY relationship, incl. non-linear & categorical
    WRAPPER   search subsets by training models   - RFE / RFECV, forward selection
    EMBEDDED  selection happens during training   - L1 (Lasso) penalties, tree importances

    And always: selection must happen INSIDE cross-validation (Module 1, Lab 11).

LEVELS
    EASY   : Correlation analysis - target correlation, redundancy, VIF, and what
             Pearson cannot see.
    MEDIUM : Mutual information - non-linear signal, the noise floor, selection in a Pipeline.
    HARD   : L1 path, RFECV, permutation importance, stability selection - and the
             business case: same AUC, a fraction of the features and data cost.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, LOAN_TARGET, RANDOM_STATE, banner, make_loan_applications, note,
                          plt, save_fig, show, step)
from preprocessing import LoanFeatureBuilder

rng = np.random.default_rng(RANDOM_STATE)

# ---------------------------------------------------------------------------
# A realistic "feature dump": useful, redundant, weak and useless columns.
# ---------------------------------------------------------------------------
loans = LoanFeatureBuilder().fit_transform(make_loan_applications(n=12000, seed=71))
y = loans[LOAN_TARGET].to_numpy()
X = pd.DataFrame({
    "payment_to_income": loans.payment_to_income,
    "loan_to_income": loans.loan_to_income,
    "credit_score": loans.credit_score.fillna(loans.credit_score.median()),
    "credit_score_missing": loans.credit_score.isna().astype(int),
    "credit_utilization": loans.credit_utilization,
    "high_utilization": loans.high_utilization,
    "delinquencies_2y": loans.delinquencies_2y,
    "age": loans.age,
    "thin_file": loans.thin_file,
    "years_employed": loans.years_employed.fillna(0),
    "annual_income": loans.annual_income,
    "loan_amount": loans.loan_amount,
    "interest_rate": loans.interest_rate,
    "num_credit_lines": loans.num_credit_lines,
    "loan_term_60": (loans.loan_term_months == 60).astype(int),
})
# Redundant copies that real feature dumps are full of
X["monthly_income"] = X.annual_income / 12                                     # exact duplicate info
X["credit_score_bureau_b"] = (X.credit_score + rng.normal(0, 15, len(X))).round()  # second bureau
# Categoricals as one-hot
for col in ["employment_type", "loan_purpose", "home_ownership", "education"]:
    X = X.join(pd.get_dummies(loans[col], prefix=col, dtype=int))
# Pure noise: what you get from joining a random marketing table
for i in range(15):
    X[f"noise_{i:02d}"] = rng.normal(size=len(X))
X["constant_flag"] = 1
TRUE_DRIVERS = {"payment_to_income", "credit_score", "credit_score_missing", "credit_utilization",
                "high_utilization", "delinquencies_2y", "age", "thin_file"}
show("Candidate features", X.shape[1])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): Correlation analysis
# ============================================================================
banner("Correlation analysis: relevance and redundancy", "EASY")

step("1.1  Remove constant columns first")
const = [c for c in Xtr.columns if Xtr[c].nunique() <= 1]
show("Constant columns", const)
Xtr, Xte = Xtr.drop(columns=const), Xte.drop(columns=const)

step("1.2  Correlation with the target: Pearson (linear) vs Spearman (rank)")
num_cols = [c for c in Xtr.columns if Xtr[c].nunique() > 2]
corr = pd.DataFrame({
    "pearson": Xtr[num_cols].corrwith(pd.Series(ytr, index=Xtr.index)),
    "spearman": Xtr[num_cols].corrwith(pd.Series(ytr, index=Xtr.index), method="spearman"),
}).assign(abs_pearson=lambda d: d.pearson.abs()).sort_values("abs_pearson", ascending=False)
print(corr.drop(columns="abs_pearson").head(12).round(3).to_string())
note("""Correlations with a 0/1 target are small in absolute terms (0.1-0.3 is
strong for credit risk). Pearson and Spearman agree for monotonic features;
Spearman is robust to the income outliers. Neither says anything about
CATEGORICAL features or U-shaped/threshold effects.""")

step("1.3  Redundancy: feature pairs that say the same thing")
cm = Xtr[num_cols].corr().abs()
pairs = (cm.where(np.triu(np.ones(cm.shape, bool), k=1)).stack()
         .sort_values(ascending=False).rename("abs corr"))
print(pairs[pairs > 0.8].round(3).to_string())
note("""monthly_income IS annual_income/12 (r = 1.0); the two bureau scores agree at
r ~0.97. Redundant features do not add information, but they split coefficient
weight unpredictably, double data costs and confuse explanations. Keep the
cheaper / more reliable one.""")

step("1.4  Variance Inflation Factor (VIF): redundancy with ALL other features")


def vif(frame):
    out = {}
    for c in frame.columns:
        r2 = LinearRegression().fit(frame.drop(columns=c), frame[c]).score(frame.drop(columns=c), frame[c])
        out[c] = 1 / max(1 - r2, 1e-9)
    return pd.Series(out).sort_values(ascending=False)


vif_cols = ["annual_income", "monthly_income", "credit_score", "credit_score_bureau_b", "interest_rate",
            "loan_amount", "payment_to_income", "loan_to_income", "age", "years_employed"]
print(vif(Xtr[vif_cols]).map(lambda v: "inf (exact copy)" if v > 1e6 else f"{v:.1f}").to_string())
note("""VIF > 10 is a common alarm level. interest_rate has a high VIF without any
exact copy: it is (almost) a linear function of credit_score - the lender priced
it that way. Multicollinearity is a problem for INTERPRETING coefficients far
more than for prediction.""")

step("1.5  What correlation cannot see")
u = rng.uniform(-3, 3, 5000)
y_u = (u ** 2 + rng.normal(0, 1, 5000) > 3).astype(int)
show("Pearson corr(u, y) for a U-shaped effect", float(np.corrcoef(u, y_u)[0, 1]))
show("Mutual information(u, y)", float(mutual_info_classif(u.reshape(-1, 1), y_u, random_state=RANDOM_STATE)[0]))
note("""Risk at BOTH extremes cancels out in a correlation. Correlation = 0 does NOT
mean 'unrelated'. Enter mutual information.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Mutual information
# ============================================================================
banner("Mutual information: any kind of dependence", "MEDIUM")
note("""MI(X; Y) = how many bits of uncertainty about Y disappear once you know X.
0 = independent. Captures non-linear and categorical relationships. sklearn
estimates it with nearest neighbours (continuous) or counts (discrete).""")
discrete = np.array([Xtr[c].nunique() <= 2 for c in Xtr.columns])
mi = pd.Series(mutual_info_classif(Xtr, ytr, discrete_features=discrete, random_state=RANDOM_STATE),
               index=Xtr.columns).sort_values(ascending=False)

step("2.1  The noise floor: MI of features that are pure noise is NOT zero")
noise_cols = [c for c in Xtr.columns if c.startswith("noise_")]
floor = mi[noise_cols].max()
show("Largest MI among 15 pure-noise features", float(floor))
shuffled = []
for s in range(5):                                # MI against a SHUFFLED target = chance level
    y_perm = np.random.default_rng(s).permutation(ytr)
    shuffled.append(mutual_info_classif(Xtr, y_perm, discrete_features=discrete, random_state=s))
threshold = np.quantile(np.concatenate(shuffled), 0.99)
show("99th percentile of MI vs shuffled target", float(threshold))
ranking = pd.DataFrame({"MI": mi, "|pearson|": Xtr.corrwith(pd.Series(ytr, index=Xtr.index)).abs()})
ranking["above noise?"] = np.where(ranking.MI > threshold, "yes", "-")
ranking["true driver?"] = ["TRUE" if c in TRUE_DRIVERS else "" for c in ranking.index]
ranking = ranking.sort_values("MI", ascending=False)
print(ranking.head(16).to_string(float_format=lambda v: f"{v:.4f}"))
note("""Estimators have variance: a noise column scores > 0 by chance. Instead of
'top-k', compare against a NULL distribution (shuffled target). Features below
that floor are indistinguishable from noise. Even so, one noise column sneaks
above a 99% threshold - with dozens of candidates, expect the occasional false
positive (another reason for stability checks).
Look at what MI MISSES: credit_score_missing and age are true drivers but fall
below the floor. MI is MARGINAL (one feature at a time): a flag that applies to
2% of applicants carries little information on average, and age's effect is
weak on its own. Filters are fast SCREENS, not verdicts - which is why the
next level uses model-based methods.""", keep_lines=False)
kept_mi = ranking.index[ranking.MI > threshold].tolist()
show("Features above the noise floor", f"{len(kept_mi)} of {Xtr.shape[1]}")

step("2.2  Selection belongs INSIDE the pipeline (fit per CV fold)")
for k in [5, 10, 20, "all"]:
    pipe = make_pipeline(StandardScaler(), SelectKBest(
        lambda X_, y_: mutual_info_classif(X_, y_, random_state=RANDOM_STATE), k=k),
        LogisticRegression(max_iter=3000))
    s = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
    show(f"CV AUC with SelectKBest(MI, k={k})", s)

# %% =========================================================================
# LEVEL 3 (HARD): Embedded & wrapper methods, stability and cost
# ============================================================================
banner("L1, RFECV, permutation importance, stability - and the business case", "HARD")
scaler = StandardScaler().fit(Xtr)
Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
cols = np.array(Xtr.columns)

step("3.1  L1 (Lasso) regularisation path: strength vs number of features")
path = []
for C in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]:
    m = LogisticRegression(l1_ratio=1, C=C, solver="liblinear").fit(Ztr, ytr)
    nz = cols[m.coef_[0] != 0]
    path.append({"C": C, "features kept": len(nz),
                 "test AUC": roc_auc_score(yte, m.predict_proba(Zte)[:, 1]),
                 "noise kept": sum(c.startswith("noise_") for c in nz)})
path = pd.DataFrame(path)
print(path.round(4).to_string(index=False))

step("3.2  RFECV: recursively drop the weakest feature, cross-validated")
rfecv = RFECV(LogisticRegression(max_iter=3000), step=2, cv=cv, scoring="roc_auc",
              min_features_to_select=3, n_jobs=-1).fit(Ztr, ytr)
show("RFECV chose", f"{rfecv.n_features_} features")
show("Noise features among them", sum(c.startswith("noise_") for c in cols[rfecv.support_]))

step("3.3  Permutation importance on held-out data (model-agnostic)")
full = LogisticRegression(max_iter=3000).fit(Ztr, ytr)
pi = permutation_importance(full, Zte, yte, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
pimp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
print(pimp.head(8).round(4).to_string())
note("""Permutation importance = 'how much does test AUC drop if I scramble this
column?'. Caveat: with redundant pairs, scrambling one copy costs little
because the other copy covers - both look unimportant. Group correlated features
(or drop duplicates first) before reading importances.""")

step("3.4  Stability selection: which features survive 50 bootstrap resamples?")
def stability_selection(features, B=50, C=0.02):
    Z = StandardScaler().fit_transform(Xtr[features])
    counts = pd.Series(0, index=features)
    for b in range(B):
        idx = np.random.default_rng(b).choice(len(Z), len(Z) // 2, replace=False)
        m = LogisticRegression(l1_ratio=1, C=C, solver="liblinear").fit(Z[idx], ytr[idx])
        counts += m.coef_[0] != 0
    return (counts / B).sort_values(ascending=False)


stability = stability_selection(list(cols))
print(stability[stability >= 0.3].round(2).to_string())
stable_raw = stability.index[stability >= 0.8].tolist()
note("""A single selection run is noisy: small data changes flip marginal features in
and out. Keep features selected in >= 80% of resamples (Meinshausen & Buhlmann,
2010). But look: BOTH bureau scores survive. With near-duplicates, L1 keeps
splitting the weight between them - selection algorithms do not understand
that two columns are 'the same thing'. That judgement is yours (step 1.3).""")

step("3.5  Remove redundant copies FIRST (decisions from 1.3), then select")
REDUNDANT = ["monthly_income", "credit_score_bureau_b", "loan_to_income"]
stability_dedup = stability_selection([c for c in cols if c not in REDUNDANT])
stable = stability_dedup.index[stability_dedup >= 0.8].tolist()
show("Stable features after de-duplication", stable)

step("3.6  The business case")


def data_cost(features):
    """Bureau A data comes as one package ($0.40); bureau B is a separate pull ($0.90)."""
    bureau_a = any(f in features for f in ["credit_score", "delinquencies_2y", "credit_utilization",
                                           "num_credit_lines", "credit_score_missing",
                                           "high_utilization", "thin_file"])
    return 0.40 * bureau_a + 0.90 * ("credit_score_bureau_b" in features)


def holdout_auc(features):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr[features], ytr)
    return roc_auc_score(yte, m.predict_proba(Xte[features])[:, 1])


options = {"All features": list(cols), "MI above noise floor": kept_mi,
           "Stability selection (raw)": stable_raw, "De-duplicate, then stability": stable}
biz = pd.DataFrame([{"feature set": k, "features": len(v), "test AUC": holdout_auc(v),
                     "data cost / 1M applications": f"${data_cost(v) * 1e6:,.0f}"}
                    for k, v in options.items()])
print(biz.round(4).to_string(index=False))
note("""Same AUC with a fraction of the inputs - and only the human de-duplication
step drops the second bureau pull, saving ~$900k per million applications.
Fewer features also means fewer things to monitor, fewer upstream dependencies
that can break, and an explanation a regulator can follow. That is the real
reason principal engineers push for feature selection.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    top = ranking.head(20)
    colors = ["#e15759" if t == "TRUE" else ("#bab0ac" if n.startswith("noise") else "#4c78a8")
              for n, t in zip(top.index, top["true driver?"])]
    axes[0].barh(top.index[::-1], top.MI[::-1], color=colors[::-1])
    axes[0].axvline(threshold, color="black", ls="--", lw=1)
    axes[0].set_title("Mutual information (red = true driver, dashed = noise floor)", fontsize=9)
    axes[1].plot(path["features kept"], path["test AUC"], marker="o")
    axes[1].set_xlabel("features kept by L1")
    axes[1].set_ylabel("test AUC")
    axes[1].set_title("L1 path: most of the AUC comes from a few features", fontsize=9)
    save_fig(fig, "07_feature_selection")

banner("KEY TAKEAWAYS")
print("""  1. Start cheap: drop constants, find redundant pairs (|r| > 0.9) and high VIF.
  2. Pearson = linear, Spearman = monotonic, MI = any dependence (incl. categorical).
  3. Compare scores against a noise floor (shuffled target) - not a fixed top-k.
  4. Select INSIDE cross-validation; prefer features that are stable across resamples.
  5. Judge a feature set on AUC AND cost, latency, monitoring burden and explainability.""")
