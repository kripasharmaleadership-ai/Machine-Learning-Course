"""
LAB 03 - Outliers and Noisy Observations
========================================

TWO DIFFERENT PROBLEMS
    OUTLIER  a RARE value far from the rest
             - ERROR      : income typed in cents (x100), age 999, utilisation 4,500%
                            -> fix it if the rule is unambiguous, otherwise set to NaN
             - GENUINE    : a surgeon earning $900k is real
                            -> keep it, but stop it dominating (cap / log / robust model)
             - INTERESTING: in fraud detection the outlier IS the target - never delete it
    NOISE    SMALL random errors on MANY values (sensor jitter, rounding, typos, label noise)
             -> cannot be "removed" row by row; reduce it (more sources, averaging,
                better measurement) and pick models that tolerate it

DETECTION TOOLKIT
    univariate : z-score (|z|>3)   IQR fences (Q1-1.5*IQR, Q3+1.5*IQR)   modified z (MAD)
    multivariate: Mahalanobis distance (robust covariance), IsolationForest
    domain rules: hard limits from the business (age 18-100, utilisation 0-150%)

HANDLING TOOLKIT
    fix (domain rule) | set NaN + impute | cap / winsorize (fit on TRAIN) | log transform |
    robust model (Huber, trees) | keep + flag

LEVELS
    EASY   : z-score vs IQR vs MAD on real (dirty) incomes - and why z-score gets "masked".
    MEDIUM : Error or genuine? Fix, cap, log, or robust model - measured on a regression.
    HARD   : Multivariate outliers (fine in every column, impossible together) and
             noisy observations (regression dilution, and how averaging sources fixes it).
"""
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_lab_utils import (HAS_MPL, LOAN_NOMINAL, LOAN_NUMERIC, LOAN_TARGET, RANDOM_STATE, banner,
                          make_loan_applications, note, plt, save_fig, show, step)
from preprocessing import Winsorizer, parse_money

rng = np.random.default_rng(RANDOM_STATE)

# Raw extract joined to the clean truth -> we KNOW which incomes are entry errors.
raw = make_loan_applications(messy=True).drop_duplicates("application_id")
truth = make_loan_applications().set_index("application_id")
raw = raw[raw.application_id.isin(truth.index)].copy()
raw["income"] = parse_money(raw.annual_income)
raw["true_income"] = raw.application_id.map(truth.annual_income)
inc = raw.dropna(subset=["income"]).copy()
inc["is_error"] = ~np.isclose(inc.income, inc.true_income, rtol=0.01)

# %% =========================================================================
# LEVEL 1 (EASY): Three detectors, one dirty column
# ============================================================================
banner("Finding outliers: z-score vs IQR vs MAD", "EASY")
x = inc.income
show("Rows with income", len(x))
show("Known entry errors (cents / monthly)", int(inc.is_error.sum()))
print(x.describe(percentiles=[.01, .5, .99]).apply(lambda v: f"{v:,.0f}").to_string())

z = (x - x.mean()) / x.std()
q1, q3 = x.quantile([.25, .75])
iqr = q3 - q1
mad = (x - x.median()).abs().median()
mod_z = 0.6745 * (x - x.median()) / mad
log_x = np.log(x)
log_mad = (log_x - log_x.median()).abs().median()
log_mod_z = 0.6745 * (log_x - log_x.median()) / log_mad

detectors = {
    "z-score |z| > 3": z.abs() > 3,
    "IQR fences (1.5 x IQR)": (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr),
    "Modified z (MAD) > 3.5": mod_z.abs() > 3.5,
    "MAD on log(income) > 3.5": log_mod_z.abs() > 3.5,
}
rows = []
for name, flag in detectors.items():
    tp = (flag & inc.is_error).sum()
    rows.append({"detector": name, "flagged": int(flag.sum()), "true errors caught": int(tp),
                 "recall": tp / inc.is_error.sum(), "precision": tp / max(flag.sum(), 1)})
print(pd.DataFrame(rows).round(2).to_string(index=False))
show("Std of income (inflated by errors)", f"${x.std():,.0f}")
note("""MASKING: a handful of huge errors (income x100) blow up the standard
deviation, so the z-score's own yardstick stretches and it flags almost
nothing. Median and MAD are not moved by extremes - robust statistics. But IQR
and MAD on the raw scale flag hundreds of GENUINE high earners (low precision),
because income is right-skewed. On the LOG scale, skew disappears and the
real errors stand out: 'income x100' becomes '+4.6 in log', a clean gap.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].hist(x, bins=80, color="#4c78a8")
    axes[0].set_title("Raw scale: one bar, everything else invisible")
    axes[0].set_xlabel("annual income ($)")
    axes[1].hist(np.log10(x[~inc.is_error]), bins=80, color="#4c78a8", label="valid")
    axes[1].hist(np.log10(x[inc.is_error]), bins=40, color="#e15759", label="entry errors")
    axes[1].set_title("log10 scale: errors form their own islands")
    axes[1].set_xlabel("log10(annual income)")
    axes[1].legend()
    save_fig(fig, "03_easy_income_outliers")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Error or genuine? And what to do about it
# ============================================================================
banner("Fix, cap, transform, or use a robust model?", "MEDIUM")

step("2.1  A decision table (write it down - it IS the business logic)")
print(pd.DataFrame([
    ["age = 0 / 999", "impossible", "-> NaN, impute + indicator", "domain rule"],
    ["utilisation > 1.5", "unit error (%)", "-> divide by 100", "domain rule, unambiguous"],
    ["income > $2M with loan < $50k", "probably cents", "-> NaN (ambiguous fix)", "domain + ratio"],
    ["income $250k-$1M", "rare but real", "-> keep; log transform / cap", "statistics"],
    ["loan_amount 10x the product max", "typo", "-> NaN, impute", "domain rule"],
    ["transaction 50x the usual (fraud)", "the signal!", "-> keep, it is the target", "use case"],
], columns=["observation", "diagnosis", "action", "basis"]).to_string(index=False))

step("2.2  Effect on a regression: predict loan_amount from income")
note("""Underwriters set the requested loan size relative to income. We fit
loan_amount ~ income on the dirty data and compare with the TRUE slope fitted on
the clean incomes. Test set = clean, unseen applicants.""")
reg = inc[["income", "true_income", "loan_amount"]].dropna()
reg = reg[reg.loan_amount < 100_000]
tr, te = train_test_split(reg, test_size=0.3, random_state=RANDOM_STATE)
Xte, yte = te[["true_income"]].to_numpy(), te.loan_amount.to_numpy()


def fit_report(name, model, X, y):
    model.fit(X, y)
    # slope measured on the model's own predictions: works for any pipeline
    p50, p100 = model.predict(np.array([[50_000.0], [100_000.0]]))
    return {"strategy": name, "slope ($ loan per $ income)": (p100 - p50) / 50_000,
            "test MAE ($)": mean_absolute_error(yte, model.predict(Xte))}


Xtr_dirty, ytr = tr[["income"]].to_numpy(), tr.loan_amount.to_numpy()
keep = (Xtr_dirty[:, 0] > 5_000) & (Xtr_dirty[:, 0] < 2_000_000)
results = [
    fit_report("Clean data (truth, for reference)", LinearRegression(), tr[["true_income"]].to_numpy(), ytr),
    fit_report("Dirty data, no treatment", LinearRegression(), Xtr_dirty, ytr),
    fit_report("Domain rule: drop impossible incomes", LinearRegression(), Xtr_dirty[keep], ytr[keep]),
    fit_report("Winsorize 1%/99% (fit on train)",
               make_pipeline(Winsorizer(0.01, 0.99), LinearRegression()), Xtr_dirty, ytr),
    fit_report("Robust model: HuberRegressor",
               make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000)), Xtr_dirty, ytr),
]
print(pd.DataFrame(results).round(4).to_string(index=False))
note("""A few dozen bad rows out of thousands flatten the least-squares slope to
almost zero: squared error gives a point 100x too far away 10,000x the
influence. Domain rules and winsorizing recover the truth. Winsorizing caps must
be LEARNED ON TRAIN and stored (Winsorizer in preprocessing.py), so production
values are capped at the same limits.
SURPRISE: the 'robust' Huber regression fails completely. Huber loss protects
against outliers in the TARGET (y). These errors are in the INPUT (x) - high-
leverage points - and Huber is not protected against those. 'Use a robust
model' is only an answer if you know WHERE the outliers are.""")

step("2.3  Does it matter for the default model? (logistic regression, test AUC)")
loans = raw.drop(columns=["annual_income", "true_income"]).rename(columns={"income": "annual_income"})
loans = loans[loans.age.between(18, 100) | loans.age.isna()]
loans["credit_utilization"] = np.where(loans.credit_utilization > 1.5,
                                       loans.credit_utilization / 100, loans.credit_utilization)
for c in LOAN_NOMINAL:
    loans[c] = loans[c].astype(str).str.strip().str.lower()
trL, teL = train_test_split(loans, test_size=0.3, stratify=loans[LOAN_TARGET], random_state=RANDOM_STATE)


def default_model(num_pipe, frame_fn=lambda f: f):
    prep = ColumnTransformer([("num", num_pipe, LOAN_NUMERIC),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), LOAN_NOMINAL)])
    tr_, te_ = frame_fn(trL), frame_fn(teL)
    m = make_pipeline(prep, LogisticRegression(max_iter=3000, C=0.5)).fit(tr_, tr_[LOAN_TARGET])
    return roc_auc_score(te_[LOAN_TARGET], m.predict_proba(te_)[:, 1])


def log_money(frame):
    out = frame.copy()
    for c in ["annual_income", "loan_amount"]:
        out[c] = np.log1p(out[c])
    return out


basic = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler())   # noqa: E731
auc = {
    "No treatment": default_model(basic()),
    "Winsorize 1/99 (in pipeline)": default_model(make_pipeline(
        SimpleImputer(strategy="median"), Winsorizer(0.01, 0.99), StandardScaler())),
    "log(income), log(loan)": default_model(basic(), log_money),
}
print(pd.Series(auc, name="test AUC").round(4).to_string())
note("""Outlier handling is not cosmetics: for a linear model, a few extreme
rows decide the scale of a whole feature. Note the same caps are applied to
the test rows by the fitted pipeline - no peeking at test quantiles.""")

# %% =========================================================================
# LEVEL 3 (HARD): Multivariate outliers and noisy observations
# ============================================================================
banner("Fine in every column, impossible together - and the cost of noise", "HARD")

step("3.1  Inject 40 'impossible combinations' - every single value looks normal")
clean = make_loan_applications(n=6000, seed=33)
cols = ["age", "years_employed", "annual_income", "loan_amount", "credit_score", "num_credit_lines"]
X = clean[cols].copy()
X["years_employed"] = X.years_employed.fillna(0)
X["credit_score"] = X.credit_score.fillna(X.credit_score.median())
b1 = rng.choice(np.where(X.age.between(28, 42))[0], 20, replace=False)
b2 = rng.choice(np.setdiff1d(np.where(X.annual_income.between(30_000, 50_000))[0], b1), 20, replace=False)
X.iloc[b1, X.columns.get_loc("years_employed")] = X.age.iloc[b1] - 12        # "working since age 12"
X.iloc[b2, X.columns.get_loc("loan_amount")] = (X.annual_income.iloc[b2] * 1.3).round(-2)  # loan > income
is_bad = np.zeros(len(X), bool)
is_bad[np.r_[b1, b2]] = True
print(X.iloc[np.r_[b1[:2], b2[:2]]].to_string())


def mad_flags(F):
    flag = np.zeros(len(F), bool)
    for c in F.columns:
        dev = F[c] - F[c].median()
        flag |= (0.6745 * dev / dev.abs().median()).abs().to_numpy() > 3.5
    return flag


def top_k(scores, k=40):
    flag = np.zeros(len(scores), bool)
    flag[np.argsort(-scores)[:k]] = True
    return flag


X_log = X.assign(annual_income=np.log(X.annual_income), loan_amount=np.log(X.loan_amount))
ratios = pd.DataFrame({"employment_ratio": X.years_employed / (X.age - 16),
                       "loan_to_income": X.loan_amount / X.annual_income})
X_ratio = pd.concat([X_log, ratios], axis=1)
S_raw, S_ratio = StandardScaler().fit_transform(X_log), StandardScaler().fit_transform(X_ratio)
methods = {
    "Per-column MAD on raw columns": mad_flags(X_log),
    "IsolationForest, raw columns (top 40)": top_k(-IsolationForest(
        n_estimators=300, random_state=RANDOM_STATE).fit(S_raw).score_samples(S_raw)),
    "Robust Mahalanobis, raw columns (top 40)": top_k(MinCovDet(random_state=RANDOM_STATE).fit(S_raw).mahalanobis(S_raw)),
    "Robust Mahalanobis + ratio features (top 40)": top_k(
        MinCovDet(random_state=RANDOM_STATE).fit(S_ratio).mahalanobis(S_ratio)),
    "Per-column MAD on the 2 ratio features": mad_flags(ratios),
    "Domain rules: years > age-16, loan > income": (
        (X.years_employed > X.age - 16) | (X.loan_amount > X.annual_income)).to_numpy(),
}
print(pd.DataFrame([{"method": k, "caught (of 40)": int((v & is_bad).sum()), "rows flagged": int(v.sum())}
                    for k, v in methods.items()]).to_string(index=False))
note("""Age 30 and 18 years of employment are both normal. Together they mean
'working since age 12'. Detectors on the raw columns, simple or fancy, miss
almost all of them. Encode the domain relationship as a RATIO feature and even
a basic per-column check finds them with no false alarms; an explicit domain
rule finds all 40. Once again, domain knowledge beats algorithm complexity.
Production pattern: rules for the known-impossible, a multivariate score as a
ranked REVIEW QUEUE for the unknown - never automatic deletion.""")

step("3.2  Noisy observations: credit score measured with error")
note("""A credit score is a noisy measurement of 'creditworthiness'. Bureaus
disagree by tens of points. What does measurement noise do to the model?""")
loans2 = make_loan_applications(n=12000, seed=34)
loans2 = loans2.dropna(subset=["credit_score"])
tr2, te2 = train_test_split(loans2, test_size=0.4, stratify=loans2[LOAN_TARGET], random_state=RANDOM_STATE)
feats = ["credit_score", "credit_utilization", "delinquencies_2y"]
rows = []
for sd in [0, 30, 60, 100]:
    def noisy(frame, k=1):
        f = frame[feats].copy()
        f["credit_score"] = f.credit_score + rng.normal(0, sd, (k, len(f))).mean(axis=0)
        return f
    for k, label in [(1, "one bureau"), (3, "average of 3 bureaus")]:
        if sd == 0 and k == 3:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression()).fit(noisy(tr2, k), tr2[LOAN_TARGET])
        coef = m[-1].coef_[0][0] / m[0].scale_[0] * 100
        rows.append({"noise sd (points)": sd, "source": label if sd else "exact",
                     "coef per 100 pts": coef,
                     "test AUC": roc_auc_score(te2[LOAN_TARGET], m.predict_proba(noisy(te2, k))[:, 1])})
noise_tbl = pd.DataFrame(rows)
print(noise_tbl.round(3).to_string(index=False))
note("""REGRESSION DILUTION: random noise in a feature biases its coefficient
towards zero (the model trusts it less) and lowers AUC. You cannot detect noise
row by row - but you CAN reduce it: average several independent sources (noise
sd drops by sqrt(3)), use better measurement, or aggregate over time. Same
idea behind ensembling and behind asking several annotators per label.""")

banner("KEY TAKEAWAYS")
print("""  1. Outliers are errors, genuine extremes, or the signal itself - diagnose before treating.
  2. z-score is masked by the outliers it hunts; use median/MAD, and log-scale skewed data.
  3. Fix only when the rule is unambiguous; otherwise NaN + impute, or cap/log/robust model.
  4. Learn caps on TRAIN, store them in the pipeline, apply the same caps in production.
  5. Multivariate detectors + ratio features catch impossible combinations; output a review queue.
  6. Noise is everywhere and small: it dilutes coefficients. Reduce it by averaging sources.""")
