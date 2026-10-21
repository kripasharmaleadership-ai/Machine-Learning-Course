"""
LAB 02 - Missing Data and Duplicate Records
===========================================

MISSING DATA - three mechanisms (Rubin, 1976). WHY a value is missing decides what you may do.
    MCAR  Missing Completely At Random   a form field randomly failed to save
          -> dropping/imputing is (mostly) harmless, just loses data
    MAR   Missing At Random              young applicants skip "years employed" more often
          (depends on OTHER observed columns) -> impute using those columns
    MNAR  Missing Not At Random          low earners refuse to state income
          (depends on the missing value ITSELF) -> imputation is biased; the fact
          that it is missing is information -> add a MISSING INDICATOR

IMPUTATION STRATEGIES (SimpleImputer)
    mean          numeric, roughly symmetric, few missing
    median        numeric, skewed or with outliers (income, prices)  <- safer default
    most_frequent categorical (mode)
    constant      "missing" as its own category / structural zero
    + add_indicator=True  -> keeps the "was missing" signal

DUPLICATES - four kinds
    exact duplicates          ETL loaded the same batch twice
    key duplicates            same ID, different content (updates / conflicts)
    near-duplicates           same entity, different spelling or ID (re-submission)
    cross-split duplicates    same entity in train AND test -> inflated test scores

LEVELS
    EASY   : Find ALL missing values (incl. disguised ones) + mean/median/mode in action.
    MEDIUM : MCAR vs MAR vs MNAR with ground truth: what imputation does to the mean,
             the spread and correlations - and to the model.
    HARD   : Duplicates: exact, conflicting, near-duplicate entity resolution and the
             cost of duplicates leaking across train/test.
"""
# %%
import difflib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_lab_utils import (HAS_MPL, LOAN_NOMINAL, LOAN_NUMERIC, LOAN_TARGET, RANDOM_STATE, banner,
                          make_loan_applications, note, plt, save_fig, show, step)
from preprocessing import MISSING_TOKENS, normalize_category, parse_dates, parse_money

rng = np.random.default_rng(RANDOM_STATE)
raw = make_loan_applications(messy=True)

# %% =========================================================================
# LEVEL 1 (EASY): Find every missing value, then impute
# ============================================================================
banner("Missing values hide in plain sight", "EASY")

step("1.1  What pandas THINKS is missing")
print(raw.isna().sum()[lambda s: s > 0].to_string())

step("1.2  What is ACTUALLY missing (tokens + sentinels + impossible values)")
SENTINELS = {"age": [0, 999, -1], "credit_score": [0]}


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        true_nan = s.isna().sum()
        token = int((s.notna() & s.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)).sum())
        sentinel = int(s.isin(SENTINELS.get(col, [])).sum())
        total = true_nan + token + sentinel
        if total:
            rows.append({"column": col, "NaN": true_nan, "text tokens ('N/A','')": token,
                         "sentinels (0, 999)": sentinel, "really missing": total,
                         "% missing": round(100 * total / len(df), 1)})
    return pd.DataFrame(rows).sort_values("really missing", ascending=False)


print(missing_report(raw).to_string(index=False))
note("""pandas reported no missing income at all - the blanks were the strings
'N/A' and ''. An age of 0 and a credit score of 0 are not values, they are
'unknown' typed by a person or a legacy system. Always convert disguised missing
values into real NaN FIRST, otherwise mean/median imputation never even runs.""")

step("1.3  Mean vs median vs mode: impute income on a small, skewed example")
df = raw.copy()
df["annual_income"] = parse_money(df.annual_income)
df["employment_type"] = normalize_category(df.employment_type)
inc = df.annual_income
show("Mean income (observed)", f"${inc.mean():,.0f}")
show("Median income (observed)", f"${inc.median():,.0f}")
note("""The mean is pulled up by a few huge incomes (and by unit errors like
income typed in cents). Median imputation gives a 'typical' applicant; mean
imputation gives every missing person an income that most applicants never
reach. For skewed money columns, MEDIAN is the safer default.""")
mode_imp = SimpleImputer(strategy="most_frequent").fit(df[["employment_type"]])
show("Mode (most_frequent) for employment_type", mode_imp.statistics_[0])

step("1.4  The side effect: imputation shrinks the spread (30% of incomes removed)")
plausible = inc[inc.between(5_000, 1_000_000)].reset_index(drop=True)     # outliers: Lab 03
holes = plausible.mask(rng.random(len(plausible)) < 0.30)
for strat in ["mean", "median"]:
    filled = pd.Series(SimpleImputer(strategy=strat).fit_transform(holes.to_frame()).ravel())
    show(f"Std after {strat} imputation",
         f"${filled.std():,.0f}  vs true ${plausible.std():,.0f}  "
         f"({filled.std() / plausible.std() - 1:+.0%}); rows sharing ONE value: "
         f"{(filled == filled.mode()[0]).mean():.0%}")
note("""Every imputed row gets the SAME value - a tall spike in the histogram,
lower variance, weaker correlations. Fine for a few %, dangerous for 30%.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): MCAR vs MAR vs MNAR - with ground truth
# ============================================================================
banner("Why the value is missing matters more than how many", "MEDIUM")
clean = make_loan_applications(n=10000, seed=21)
true_inc = clean.annual_income.to_numpy()
age = clean.age.to_numpy()

masks = {
    "MCAR (random 30%)": rng.random(len(clean)) < 0.30,
    "MAR (younger skip more)": rng.random(len(clean)) < np.clip(0.75 - 0.012 * (age - 18), 0.02, 0.9),
    "MNAR (low earners skip)": rng.random(len(clean)) < np.clip(1.15 - true_inc / 60000, 0.02, 0.95),
}
rows = []
for name, m in masks.items():
    obs = pd.Series(np.where(m, np.nan, true_inc))
    med = obs.fillna(obs.median())
    reg = IterativeImputer(random_state=RANDOM_STATE).fit_transform(
        np.column_stack([obs, age, clean.loan_amount, clean.years_employed.fillna(0)]))[:, 0]
    rows.append({"mechanism": name, "% missing": round(100 * m.mean(), 1),
                 "true mean": true_inc.mean(), "mean of observed": obs.mean(),
                 "median-imputed mean": med.mean(), "iterative-imputed mean": reg.mean(),
                 "true corr(income,loan)": np.corrcoef(true_inc, clean.loan_amount)[0, 1],
                 "corr after median imp.": np.corrcoef(med, clean.loan_amount)[0, 1]})
mech = pd.DataFrame(rows).set_index("mechanism")
print(mech.round(3).to_string())
note("""MCAR: the observed data is still a fair sample - only precision is lost.
MAR: bias exists, but the CAUSE (age) is observed, so a model-based imputer
(IterativeImputer uses the other columns) removes much of it.
MNAR: the observed incomes are systematically too high, and NO imputer can see
the reason - it is the missing value itself. Every simple imputation also
weakens the correlation with loan_amount, which hurts the model.""")

step("2.1  In our loan data credit_score is MNAR: thin-file applicants have no score")
loans = make_loan_applications(n=10000, seed=31)
tr, te = train_test_split(loans, test_size=0.4, stratify=loans[LOAN_TARGET], random_state=RANDOM_STATE)
show("Default rate when credit_score is present", f"{tr[tr.credit_score.notna()][LOAN_TARGET].mean():.1%}")
show("Default rate when credit_score is MISSING", f"{tr[tr.credit_score.isna()][LOAN_TARGET].mean():.1%}")

step("2.2  Six ways to handle it - same model, same split")
NUM = LOAN_NUMERIC


def evaluate(num_imputer, drop_rows=False, model="logistic"):
    train = tr.dropna(subset=["credit_score"]) if drop_rows else tr
    clf = (LogisticRegression(max_iter=3000, C=0.5) if model == "logistic"
           else HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    steps = [num_imputer] if num_imputer is not None else []
    if model == "logistic":
        steps.append(StandardScaler())
    prep = ColumnTransformer([("num", make_pipeline(*steps) if steps else "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), LOAN_NOMINAL)],
                             sparse_threshold=0)
    pipe = make_pipeline(prep, clf).fit(train, train[LOAN_TARGET])
    # In production you cannot drop an applicant - the pipeline's imputer still fills them.
    p = pipe.predict_proba(te)[:, 1]
    no_score = te.credit_score.isna().to_numpy()
    return {"test AUC": roc_auc_score(te[LOAN_TARGET], p),
            "predicted default, NO-score applicants": p[no_score].mean()}


strategies = {
    "Drop rows with missing score": evaluate(SimpleImputer(strategy="median"), drop_rows=True),
    "Mean imputation": evaluate(SimpleImputer(strategy="mean")),
    "Median imputation": evaluate(SimpleImputer(strategy="median")),
    "Median + missing indicator": evaluate(SimpleImputer(strategy="median", add_indicator=True)),
    "IterativeImputer (model-based)": evaluate(IterativeImputer(random_state=RANDOM_STATE)),
    "Boosting, native NaN handling": evaluate(None, model="hgb"),
}
res = pd.DataFrame(strategies).T
print(res.round(3).to_string())
show("ACTUAL default rate, NO-score applicants (test)", f"{te[te.credit_score.isna()][LOAN_TARGET].mean():.3f}")
note("""Overall AUC barely moves - only ~2% of applicants lack a score, so a global
metric hides the damage. Look at the segment instead: the blank IS the signal
('thin credit file'), and thin files default more. Dropping rows or imputing a
median tells the model 'this is an average applicant' -> their risk is
underestimated by a third. The missing indicator (one extra 0/1 column) keeps
the fact. IterativeImputer also does well HERE because another column
(num_credit_lines) predicts the missing score - a MAR-like escape hatch that
does not always exist. Boosting learns its own route for NaN.
Lesson: always evaluate imputation choices on the AFFECTED segment.""")

if HAS_MPL:
    m = masks["MNAR (low earners skip)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bins = np.linspace(0, 250000, 60)
    axes[0].hist(true_inc, bins=bins, alpha=0.5, label="true income (all)")
    axes[0].hist(true_inc[~m], bins=bins, alpha=0.6, label="observed (MNAR)")
    axes[0].set_title("MNAR: observed incomes are not a fair sample")
    axes[0].legend()
    obs = np.where(m, np.nan, true_inc)
    axes[1].hist(np.where(m, np.nanmedian(obs), obs), bins=bins, color="#e15759")
    axes[1].set_title("After median imputation: one huge spike")
    for ax in axes:
        ax.set_xlabel("annual income ($)")
    save_fig(fig, "02_medium_mnar_imputation")

# %% =========================================================================
# LEVEL 3 (HARD): Duplicates and entity resolution
# ============================================================================
banner("Duplicates: exact, conflicting, near - and leaking", "HARD")

step("3.1  Exact duplicates (ETL loaded a batch twice)")
show("Exact duplicate rows", raw.duplicated().sum())
show("Duplicate application_id values", raw.application_id.duplicated().sum())
dedup = raw.drop_duplicates().copy()

step("3.2  Conflicting duplicates: same ID, different values")
updates = dedup.sample(60, random_state=1).copy()
updates["credit_score"] = updates.credit_score + rng.integers(-30, 31, len(updates))
updates["record_updated_at"] = "2026-01-15"
dedup["record_updated_at"] = "2025-12-31"
with_updates = pd.concat([dedup, updates], ignore_index=True)
conflicts = with_updates[with_updates.application_id.duplicated(keep=False)]
show("IDs with conflicting versions", conflicts.application_id.nunique())
print(conflicts.sort_values("application_id")[["application_id", "credit_score",
                                                 "record_updated_at"]].head(6).to_string(index=False))
resolved = (with_updates.sort_values("record_updated_at")
            .drop_duplicates("application_id", keep="last"))      # survivorship rule: latest wins
show("Rows after 'latest record wins'", len(resolved))
note("""drop_duplicates() would NOT remove these (the rows differ). You need a
SURVIVORSHIP RULE: latest timestamp wins, or most trusted source wins, or
field-level merge. Write the rule down - it is a business decision.""")

step("3.3  Near-duplicates: same person re-submitted under a NEW application_id")
d = dedup.drop(columns="record_updated_at").copy()
d["employer_norm"] = normalize_category(d.employer)
d["income_num"] = parse_money(d.annual_income)
d["date"] = parse_dates(d.application_date)
key = ["age", "income_num", "loan_amount", "loan_purpose", "date", "employer_norm"]
near = d[d.duplicated(subset=key, keep=False)].sort_values(key)
show("Rows in near-duplicate groups", len(near))
print(near[["application_id", "employer", "annual_income", "purpose_text"]].head(6).to_string(index=False))
found = set(near.application_id)
truly_resubmitted = set(d.application_id[d.application_id.str.startswith("APP-9")])
show("Re-submissions found / actually present",
     f"{len(found & truly_resubmitted)} / {len(truly_resubmitted)}")
note("""Without normalisation ('UNION ACME HEALTH' vs 'Union Acme Health') the
key never matches. Real entity resolution adds BLOCKING (only compare rows
that share e.g. date + state) and FUZZY similarity, because exact keys miss
typos. A taste of fuzzy matching:""")
for a, b in [("Union Acme Health", "UNION  ACME HEALTH"), ("Union Acme Health", "Union Acme Bank"),
             ("Summit Wonka Foods", "Sumit Wonka Food")]:
    raw_sim = difflib.SequenceMatcher(None, a, b).ratio()
    norm_sim = difflib.SequenceMatcher(None, " ".join(a.lower().split()), " ".join(b.lower().split())).ratio()
    print(f"     {a!r:22} vs {b!r:22}  raw={raw_sim:.2f}  normalised={norm_sim:.2f}")

step("3.4  Why it matters: duplicates leaking across train and test")
leak = d.drop(columns=["employer_norm", "income_num", "date"]).copy()
leak["annual_income"] = parse_money(leak.annual_income)
leak = pd.concat([leak, leak.sample(frac=0.25, random_state=2)], ignore_index=True)   # 25% re-loaded
tr2, te2 = train_test_split(leak, test_size=0.3, random_state=RANDOM_STATE)
num = ["age", "annual_income", "credit_score", "num_credit_lines", "credit_utilization",
       "delinquencies_2y", "loan_amount", "interest_rate"]
memorizer = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(300, min_samples_leaf=1, n_jobs=-1,
                                                 random_state=RANDOM_STATE)).fit(tr2[num], tr2[LOAN_TARGET])
seen = te2.application_id.isin(tr2.application_id)
p = memorizer.predict_proba(te2[num])[:, 1]
show("Test rows that also exist in train", f"{seen.mean():.1%}")
show("AUC on those 'seen' test rows", roc_auc_score(te2[LOAN_TARGET][seen], p[seen]))
show("AUC on genuinely new test rows", roc_auc_score(te2[LOAN_TARGET][~seen], p[~seen]))
show("Reported AUC (all test rows mixed)", roc_auc_score(te2[LOAN_TARGET], p))
note("""A flexible model memorises the training rows; when the same rows show up
in test, it 'predicts' them perfectly. The reported AUC is a blend of real skill
and memory. Rule: DEDUPLICATE (and group by entity) BEFORE splitting.""")

banner("KEY TAKEAWAYS")
print("""  1. Convert disguised missing values ('N/A', '', 0, 999) to NaN before anything else.
  2. Median for skewed numbers, mode for categories; every imputation shrinks variance.
  3. Ask WHY data is missing: MCAR is harmless, MAR is fixable, MNAR needs an indicator.
  4. Missingness is often a feature: add_indicator=True is cheap insurance.
  5. Deduplicate exact rows, resolve conflicts with a written rule, normalise before matching.
  6. Deduplicate before the train/test split, or your test score includes memory.""")
