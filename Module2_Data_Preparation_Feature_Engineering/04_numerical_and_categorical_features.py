"""
LAB 04 - Numerical and Categorical Features (and how to encode them)
====================================================================

FEATURE TYPES (decide the type from MEANING, not from the dtype pandas guessed)
    NUMERICAL
        continuous   income, interest_rate            - arithmetic makes sense
        discrete     num_credit_lines, delinquencies  - counts
    CATEGORICAL
        nominal      state, loan_purpose              - no order   ("TX" is not > "CA")
        ordinal      education                        - order, but no distances
        binary       has_cosigner yes/no
    OTHER
        datetime     application_date  -> engineer parts (Lab 06)
        free text    purpose_text      -> tokens / embeddings (Lab 11)
        identifier   application_id    -> never a feature
    TRAPS
        numbers that are categories  : zip code, product code, loan_term_months (36 / 60)
        categories that are numbers  : "1-2 years", "Low/Medium/High"

ENCODING
    One-hot  (OneHotEncoder)  nominal -> one 0/1 column per category. Safe for every model.
    Ordinal  (OrdinalEncoder) ordinal -> 0,1,2,... in the RIGHT order (you must give the order!)
    High cardinality (1,000+ levels): min_frequency, frequency, hashing, target encoding.

LEVELS
    EASY   : Type every column; the "state as a number" trap.
    MEDIUM : One-hot vs ordinal - the alphabetical-order bug, linear vs tree models,
             unseen categories in production, rare categories.
    HARD   : 1,200 employers: six encodings compared on accuracy, width and robustness.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler,
                                   TargetEncoder)

from ml_lab_utils import (EDUCATION_LEVELS, HAS_MPL, LOAN_TARGET, RANDOM_STATE, banner,
                          make_loan_applications, note, plt, save_fig, show, step)
from preprocessing import LoanFeatureBuilder

loans = make_loan_applications(n=12000, seed=41)
loans = LoanFeatureBuilder().fit_transform(loans)
tr, te = train_test_split(loans, test_size=0.3, stratify=loans[LOAN_TARGET], random_state=RANDOM_STATE)
y_tr, y_te = tr[LOAN_TARGET], te[LOAN_TARGET]
BASE_NUM = ["payment_to_income", "credit_score", "credit_utilization", "delinquencies_2y",
            "age", "num_credit_lines"]

# %% =========================================================================
# LEVEL 1 (EASY): What type is each column?
# ============================================================================
banner("Feature types: meaning beats dtype", "EASY")
types = pd.DataFrame([
    ["application_id", "identifier", "drop - unique per row, carries no pattern"],
    ["application_date", "datetime", "engineer: month, weekday, recency (Lab 06)"],
    ["age", "numerical - continuous", "scale; maybe non-linear (young = risky)"],
    ["employment_type", "categorical - nominal", "one-hot"],
    ["years_employed", "numerical - continuous", "structural NaN for unemployed"],
    ["employer", "categorical - nominal, HIGH cardinality", "~1,200 levels -> see HARD"],
    ["education", "categorical - ORDINAL", "HS < Bachelor < Master < PhD"],
    ["annual_income", "numerical - continuous, skewed", "log / robust scale (Lab 05)"],
    ["state", "categorical - nominal", "one-hot (15 levels)"],
    ["num_credit_lines", "numerical - discrete (count)", "as number"],
    ["loan_term_months", "numerical dtype, but 2 values", "treat as binary category (36 vs 60)"],
    ["purpose_text", "free text", "tokens / embeddings (Lab 11)"],
    ["defaulted", "TARGET", "never a feature"],
], columns=["column", "type (by MEANING)", "treatment"])
print(types.to_string(index=False))
show("pandas dtype of loan_term_months", str(loans.loan_term_months.dtype))
show("unique values", sorted(loans.loan_term_months.unique()))

step("1.1  The trap: give each loan purpose a number (business=0, car=1, ...)")
codes = {s: i for i, s in enumerate(sorted(loans.loan_purpose.unique()))}
rate = tr.groupby("loan_purpose")[LOAN_TARGET].mean()
print(pd.DataFrame({"code": pd.Series(codes), "default rate": rate}).round(3).to_string())
note("""A linear model with 'purpose_code' learns ONE slope: 'each step down the
alphabet changes risk by X'. That relationship does not exist - business (0)
and wedding (7) are both risky, car (1) is safe. One-hot gives each purpose its
own coefficient.""")
auc_code = roc_auc_score(y_te, make_pipeline(StandardScaler(), LogisticRegression()).fit(
    tr.loan_purpose.map(codes).to_frame(), y_tr).predict_proba(te.loan_purpose.map(codes).to_frame())[:, 1])
auc_ohe = roc_auc_score(y_te, make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression()).fit(
    tr[["loan_purpose"]], y_tr).predict_proba(te[["loan_purpose"]])[:, 1])
show("AUC from loan_purpose alone - as a number", auc_code)
show("AUC from loan_purpose alone - one-hot", auc_ohe)
note("""Below 0.5 = worse than a coin flip: the fake order is actively misleading.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): One-hot vs ordinal encoding
# ============================================================================
banner("One-hot vs ordinal: getting the details right", "MEDIUM")

step("2.1  OrdinalEncoder's default order is ALPHABETICAL")
default_enc = OrdinalEncoder().fit(tr[["education"]])
right_enc = OrdinalEncoder(categories=[EDUCATION_LEVELS]).fit(tr[["education"]])
print(pd.DataFrame({"default (alphabetical)": default_enc.categories_[0],
                    "explicit order": right_enc.categories_[0]}).to_string())
note("""Default: Bachelor=0, High School=1, Master=2, PhD=3 - High School now sits
ABOVE Bachelor. No error, no warning. Always pass categories=[...] explicitly.""")

step("2.2  Encoding choice x model family (test AUC)")
cat_nominal = ["employment_type", "state", "loan_purpose", "home_ownership"]


def run(encoding, model):
    if encoding == "one-hot nominal + ordered education":
        enc = [("nom", OneHotEncoder(handle_unknown="ignore"), cat_nominal),
               ("edu", OrdinalEncoder(categories=[EDUCATION_LEVELS]), ["education"])]
    elif encoding == "ordinal codes for everything (alphabetical)":
        enc = [("all", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_nominal + ["education"])]
    else:  # one-hot everything
        enc = [("all", OneHotEncoder(handle_unknown="ignore"), cat_nominal + ["education"])]
    prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                               BASE_NUM)] + enc, sparse_threshold=0)
    clf = (LogisticRegression(max_iter=3000) if model == "Logistic regression"
           else HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    p = make_pipeline(prep, clf).fit(tr, y_tr).predict_proba(te)[:, 1]
    return roc_auc_score(y_te, p)


encodings = ["one-hot nominal + ordered education", "one-hot everything",
             "ordinal codes for everything (alphabetical)"]
grid = pd.DataFrame({m: [run(e, m) for e in encodings] for m in ["Logistic regression", "Gradient boosting"]},
                    index=encodings)
print(grid.round(4).to_string())
note("""Linear models need one-hot for nominal data: arbitrary codes impose a
fake order and the model cannot undo it. Trees can split codes into any groups,
so they tolerate ordinal codes (and it keeps the data narrow). For ordinal
features one-hot is also fine - it just ignores the order information.""")

step("2.3  Production day 1: a category the model has never seen")
new_app = te.iloc[[0]].copy()
new_app["employment_type"] = "Gig worker"                 # added to the web form last week
strict = OneHotEncoder(handle_unknown="error").fit(tr[["employment_type"]])
try:
    strict.transform(new_app[["employment_type"]])
except ValueError as e:
    show("handle_unknown='error'", f"CRASH -> {str(e)[:60]}...")
lenient = OneHotEncoder(handle_unknown="ignore").fit(tr[["employment_type"]])
show("handle_unknown='ignore'", lenient.transform(new_app[["employment_type"]]).toarray().tolist())
note("""'ignore' encodes the unknown as all zeros = 'like the reference/average
category'. It keeps the service up - but MONITOR the unknown rate: a sudden
jump means the upstream form changed and the model needs retraining.""")

step("2.4  Rare categories: min_frequency groups them into 'infrequent'")
rare = OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist").fit(tr[["home_ownership", "loan_purpose"]])
print(f"  output columns: {list(rare.get_feature_names_out())}")
note("""Categories seen fewer than 50 times get one shared column. Their own
coefficients would be pure noise anyway, and new/unknown values fall into the
same bucket in production.""")

# %% =========================================================================
# LEVEL 3 (HARD): High cardinality - 1,200 employers
# ============================================================================
banner("1,200 employers: which encoding?", "HARD")
emp_tr = tr.employer.fillna("none")
show("Distinct employers in training", emp_tr.nunique())
show("Employers seen fewer than 5 times", f"{(emp_tr.value_counts() < 5).mean():.0%}")
show("Test rows with an employer unseen in training",
     f"{(~te.employer.fillna('none').isin(set(emp_tr))).mean():.1%}")


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace a category with how often it appeared in TRAINING (popular employers)."""

    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X).ravel())
        self.freq_ = s.value_counts(normalize=True)
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        return s.map(self.freq_).fillna(0).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["employer_freq"])


hasher = make_pipeline(
    FunctionTransformer(lambda X: np.asarray(X).astype(str)),
    FeatureHasher(n_features=128, input_type="string"))
fill = SimpleImputer(strategy="constant", fill_value="none")
ENCODERS = {
    "Drop the column": None,
    "One-hot (all levels)": make_pipeline(fill, OneHotEncoder(handle_unknown="ignore")),
    "One-hot, min_frequency=20": make_pipeline(fill, OneHotEncoder(min_frequency=20,
                                                                   handle_unknown="infrequent_if_exist")),
    "Frequency encoding": make_pipeline(fill, FrequencyEncoder()),
    "Hashing trick (128 buckets)": make_pipeline(fill, hasher),
    "Target encoding (cross-fitted)": make_pipeline(fill, TargetEncoder(
        smooth=20, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE))),
}
rows = []
for name, enc in ENCODERS.items():
    parts = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), BASE_NUM),
             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_nominal)]
    if enc is not None:
        parts.append(("employer", enc, ["employer"]))
    pipe = make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=5000, C=0.5))
    t0 = time.perf_counter()
    pipe.fit(tr, y_tr)
    fit_s = time.perf_counter() - t0
    width = pipe[0].transform(tr.head(5)).shape[1]
    rows.append({"encoding": name, "features": width, "test AUC": roc_auc_score(y_te, pipe.predict_proba(te)[:, 1]),
                 "fit seconds": fit_s})
hc = pd.DataFrame(rows)
print(hc.round(4).to_string(index=False))
note("""Employer carries real signal (dropping it costs AUC). Then compare WIDTH:
 - Full one-hot wins here, because logistic regression's L2 penalty shrinks
   rare-employer coefficients toward zero - a built-in safety net. But it is
   ~1,000 columns, and the vocabulary must be maintained and versioned.
 - min_frequency=20 gets almost the same AUC with ~12x fewer columns.
 - Hashing: fixed width, no vocabulary to store; collisions are the price.
 - Target encoding: ONE column. Slightly lower for a linear model here, but it
   is the go-to for trees and for MILLIONS of levels (user IDs, URLs) where
   one-hot is impossible. It must be cross-fitted - TargetEncoder does that
   inside fit_transform - or each row's own label leaks into its feature.
There is no universal winner. Measure on YOUR data, and include width,
maintenance and serving cost in the decision.""", keep_lines=True)

step("3.1  What target encoding learned (training data, smoothed)")
te_enc = TargetEncoder(smooth=20).fit(emp_tr.to_frame(), y_tr)
learned = pd.Series(te_enc.encodings_[0], index=te_enc.categories_[0])
counts = emp_tr.value_counts()
view = pd.DataFrame({"applicants": counts, "raw default rate": y_tr.groupby(emp_tr).mean(),
                     "smoothed encoding": learned}).dropna()
print(pd.concat([view.sort_values("applicants", ascending=False).head(4),
                 view[view.applicants == 1].head(3)]).round(3).to_string())
note("""Large employers keep close to their own rate. An employer seen ONCE is
pulled almost all the way to the global mean (~0.19) - smoothing stops one
lucky or unlucky applicant from white- or blacklisting a whole company.
Careful: smooth='auto' (the default) does NOT shrink categories seen once,
because their within-category variance is zero. Set smooth explicitly.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.scatter(hc.features, hc["test AUC"], s=60, color="#4c78a8")
    for _, r in hc.iterrows():
        ax.annotate(r.encoding, (r.features, r["test AUC"]), fontsize=8, xytext=(5, 3),
                    textcoords="offset points")
    ax.set_xscale("symlog")
    ax.set_xlabel("number of model input features (log scale)")
    ax.set_ylabel("test AUC")
    ax.set_title("High-cardinality encodings: width vs accuracy")
    save_fig(fig, "04_hard_high_cardinality")

banner("KEY TAKEAWAYS")
print("""  1. Type columns by MEANING: zip codes and terms are categories; IDs are never features.
  2. Nominal -> one-hot. Ordinal -> OrdinalEncoder WITH an explicit order (default is alphabetical).
  3. Trees tolerate ordinal codes; linear models and neural nets do not.
  4. Plan for production: handle_unknown, min_frequency, and monitor the unknown rate.
  5. High cardinality: min_frequency, hashing, or cross-fitted target encoding - never naive mean encoding.""")
