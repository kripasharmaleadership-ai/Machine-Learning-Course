"""
LAB 12 - ASSIGNMENT WORKED EXAMPLE: Prepare a Raw Dataset and Document Every Decision
=====================================================================================

THE ASSIGNMENT (see ASSIGNMENT.md)
    "Prepare a raw dataset and document every preprocessing decision."
    This lab is the REFERENCE SOLUTION on the raw loan-applications extract.
    Your assignment applies the same method to a different raw dataset
    (used-car listings: assignment_starter.py).

WHAT "DOCUMENT EVERY DECISION" MEANS IN INDUSTRY
    For each decision a reviewer should be able to read:
        WHAT  was done, to WHICH columns, affecting HOW MANY rows
        WHY   (the evidence - a number, not an opinion)
        WHAT ELSE was considered, and why it was rejected
    Deliverables: cleaned data + decision log + data card + a reproducible pipeline.

LEVELS
    EASY   : Profile the raw data; see what a decision-log entry looks like.
    MEDIUM : Clean step by step - every step measures itself and writes to the log.
    HARD   : Prove it: contract checks pass, before/after comparison, leakage review,
             time-based split, model check, and export all deliverables.
"""
# %%
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from ml_lab_utils import (EDUCATION_LEVELS, LOAN_NOMINAL, LOAN_NUMERIC, LOAN_TARGET, OUT_DIR, banner,
                          make_loan_applications, note, show, step)
from preprocessing import (ColumnRule, DataContract, DecisionLog, LoanFeatureBuilder, LoanRawCleaner,
                           build_preprocessor, fingerprint, normalize_category, parse_dates, parse_money)

raw = make_loan_applications(messy=True)
log = DecisionLog("Personal-loan applications (raw production extract)")

# %% =========================================================================
# LEVEL 1 (EASY): Profile, then see what an entry looks like
# ============================================================================
banner("Profile the raw data", "EASY")


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        as_num = pd.to_numeric(s, errors="coerce")
        rows.append({"column": c, "dtype": str(s.dtype), "missing %": round(100 * s.isna().mean(), 1),
                     "blank/N/A strings": int(s.astype(str).str.strip().str.lower().isin(["", "n/a"]).sum()),
                     "unique": s.nunique(),
                     "numeric-parsable %": round(100 * as_num.notna().mean(), 1),
                     "example": s.dropna().astype(str).iloc[0][:28]})
    return pd.DataFrame(rows)


print(profile(raw).to_string(index=False))
show("Rows (raw)", len(raw))
show("Data fingerprint (raw)", fingerprint(raw))

step("An example decision-log entry")
example = DecisionLog("example")
example.record("missing values", "credit_score", "median imputation + missing indicator",
               "Missingness is informative: default rate 24% when missing vs 16% when present",
               "drop rows (loses the riskiest segment); mean (skew)", "see Lab 02, section 2.2", 125)
print(example.to_frame().T.to_string(header=False))

# %% =========================================================================
# LEVEL 2 (MEDIUM): Clean step by step - each step measures itself
# ============================================================================
banner("Clean step by step, logging every decision", "MEDIUM")
df = raw.copy()


def decide(*args, **kwargs):
    """Record a decision AND echo it, so the log is built in plain sight."""
    log.record(*args, **kwargs)
    e = log.entries[-1]
    rows = f" | rows: {e['rows_affected']:,}" if e["rows_affected"] is not None else ""
    ev = f" | {e['evidence']}" if e["evidence"] else ""
    print(f"  [#{e['#']:>2}] {e['columns'][:30]:<30} {e['decision'][:60]}{rows}{ev}")

step("Step 1 - Exact duplicates")
n = df.duplicated().sum()
df = df.drop_duplicates()
decide("uniqueness", "all", "drop exact duplicate rows", "ETL reloads: identical rows incl. application_id",
           "keep (would overweight those applicants and leak across splits)", f"{n} exact duplicates", n)

step("Step 2 - Parse types")
before_na = df.application_date.isna().sum()
df["application_date"] = parse_dates(df.application_date)
bad_dates = df.application_date.isna().sum() - before_na
decide("parsing", "application_date", "parse with format='mixed' (ISO + US formats)",
           "Two upstream systems write different formats", "drop non-ISO rows (20% of data)",
           f"{bad_dates} unparseable after parsing", int(bad_dates))
n_str = df.annual_income.astype(str).str.contains(r"\$").sum()
n_na = df.annual_income.astype(str).str.strip().str.lower().isin(["n/a", ""]).sum()
df["annual_income"] = parse_money(df.annual_income)
decide("parsing", "annual_income", "strip '$' and ',' -> float; 'N/A' and '' -> NaN",
           "Currency-formatted strings; pd.to_numeric would silently turn them into NaN",
           "pd.to_numeric(errors='coerce')", f"{n_str} '$' strings, {n_na} blank/N/A", int(n_str + n_na))

step("Step 3 - Sentinels and impossible values -> NaN")
for col, lo, hi, why in [("age", 18, 100, "0 / 999 are 'unknown' codes"),
                         ("credit_score", 300, 850, "0 is the legacy 'no score' code")]:
    bad = df[col].notna() & ~df[col].between(lo, hi)
    df.loc[bad, col] = np.nan
    decide("validity", col, f"values outside [{lo}, {hi}] -> NaN", why, "clip to range (invents values)",
               f"values found: {[float(v) for v in sorted(raw[col].dropna()[~raw[col].dropna().between(lo, hi)].unique())[:3]]}",
               int(bad.sum()))

step("Step 4 - Unit errors")
pct = df.credit_utilization > 1.5
df.loc[pct, "credit_utilization"] /= 100
decide("validity", "credit_utilization", "values > 1.5 divided by 100",
           "Entered as a percentage; utilisation above 150% does not occur, and /100 lands in the normal range",
           "NaN + impute", f"max before {raw.credit_utilization.max():.0f}, after {df.credit_utilization.max():.2f}",
           int(pct.sum()))
big_inc = df.annual_income > 2_000_000
df.loc[big_inc, "annual_income"] = np.nan
decide("validity", "annual_income", "> $2M -> NaN (then imputed)",
           "Consistent with income typed in cents (x100); the true value is ambiguous, so do not 'fix'",
           "divide by 100 (guess); keep (destroys linear models, Lab 03)",
           f"{big_inc.sum()} values, max {raw.annual_income.pipe(parse_money).max():,.0f}", int(big_inc.sum()))
big_loan = df.loan_amount > 75_000
df.loc[big_loan, "loan_amount"] = np.nan
decide("validity", "loan_amount", "> product maximum $75k -> NaN", "Product policy caps loans at $75k; "
           "values are exact x10 typos", "divide by 10", f"{big_loan.sum()} values", int(big_loan.sum()))

step("Step 5 - Consistent categories")
before = df.employment_type.nunique()
df["employment_type"] = normalize_category(df.employment_type, LoanRawCleaner.EMPLOYMENT)
df["home_ownership"] = normalize_category(df.home_ownership, LoanRawCleaner.HOME)
df["state"] = df.state.str.strip().str.upper()
df["employer"] = normalize_category(df.employer).str.title()
df["purpose_text"] = normalize_category(df.purpose_text)
print(df.employment_type.value_counts(dropna=False).to_string())
decide("consistency", ["employment_type", "home_ownership", "state", "employer", "purpose_text"],
           "trim, lower-case, unify separators, map synonyms (e.g. 'Contractor' -> 'Contract'; "
           "'OTHER'/'NONE' -> 'Other')", "Free-text entry produced spelling variants of the same category",
           "one-hot as-is (17 fake categories)", f"employment_type: {before} spellings -> "
           f"{df.employment_type.nunique()} categories")

step("Step 6 - Near-duplicates (re-submissions under a new ID)")
key = ["age", "annual_income", "loan_amount", "loan_purpose", "application_date", "employer", "state"]
near = df.duplicated(subset=key, keep="first")
df = df[~near]
decide("uniqueness", key, "drop later re-submissions with identical applicant + loan fields",
           "Same person re-applied; keeping both double-counts them and leaks across splits",
           "fuzzy matching (not needed: exact match after normalisation)", f"{near.sum()} re-submissions",
           int(near.sum()))

step("Step 7 - Missing values: strategy per column (applied later, inside the pipeline)")
miss = df[LOAN_NUMERIC + LOAN_NOMINAL + ["education", "employer"]].isna().mean().round(3)
print(miss[miss > 0].rename("share missing").to_string())
decide("missing values", "years_employed", "median + missing indicator",
           "Structurally missing for the unemployed (not applicable); indicator keeps that fact", "fill 0")
decide("missing values", ["annual_income", "credit_score", "age"], "median + missing indicator",
           "MNAR: low earners skip income; thin files have no score (Lab 02)", "drop rows; mean",
           f"income {miss['annual_income']:.1%}, score {miss['credit_score']:.1%}, age {miss['age']:.1%} missing")
decide("missing values", ["home_ownership", "employment_type", "state", "loan_purpose"],
           "constant 'missing' category before one-hot", "Few rows; keeps them scoreable", "mode")
decide("feature selection", "employer", "excluded from v1", "~1,200 levels; needs target encoding "
           "(Lab 04) - deferred to v2 to keep v1 simple and auditable", "one-hot; target encoding")

# %% =========================================================================
# LEVEL 3 (HARD): Prove it, then ship it
# ============================================================================
banner("Validate, review leakage, split, model-check, export", "HARD")

step("3.1  Re-run the data contract on the cleaned data")
contract = DataContract(
    rules={"age": ColumnRule("number", 18, 100, max_missing=0.05),
           "annual_income": ColumnRule("number", 5_000, 2_000_000, max_missing=0.10),
           "credit_score": ColumnRule("number", 300, 850, max_missing=0.05),
           "credit_utilization": ColumnRule("number", 0, 1.5),
           "loan_amount": ColumnRule("number", 500, 75_000, max_missing=0.01),
           "employment_type": ColumnRule("category", allowed=["Salaried", "Self-employed", "Contract",
                                                              "Unemployed", "Retired"]),
           "home_ownership": ColumnRule("category", allowed=["Rent", "Mortgage", "Own", "Other"])},
    unique_key="application_id")
show("Contract violations - raw", len(contract.validate(raw)))
issues = contract.validate(df)
show("Contract violations - after steps 1-7", len(issues))
print(issues.to_string(index=False))
note("""The contract caught something our step-by-step cleaning MISSED: incomes
below $5,000 a year. They look like MONTHLY incomes typed into the annual field
(x12 would be plausible) - the mirror image of the 'cents' error. This is why
the contract runs AFTER cleaning too. Fix, log, re-validate:""")
low_inc = df.annual_income < 5_000
df.loc[low_inc, "annual_income"] = np.nan
decide("validity", "annual_income", "< $5,000 -> NaN (then imputed)",
       "Consistent with monthly income typed into the annual field; x12 is a guess, so do not 'fix'",
       "multiply by 12 (guess); keep", f"caught by the data contract after cleaning; min was "
       f"{raw.annual_income.pipe(parse_money).min():,.0f}", int(low_inc.sum()))
issues = contract.validate(df)
show("Contract violations - after the fix", len(issues))

step("3.2  Leakage & fairness review (a human checklist, recorded)")
review = [
    ("interest_rate", "KEEP, flagged", "Set by the CURRENT underwriting model from the credit score: known at "
     "decision time, but learning from the old model's output can freeze its biases. Revisit in v2."),
    ("purpose_text", "EXCLUDE from v1", "Free text duplicates loan_purpose; may contain PII. Candidate for "
     "embeddings (Lab 11) after a PII review."),
    ("application_id", "EXCLUDE", "Identifier"),
    ("state", "KEEP, monitor", "Geography can proxy protected attributes; check approval rates by state."),
    ("age", "KEEP, legal review", "Protected attribute in many jurisdictions (e.g. ECOA in the US) - legal "
     "sign-off required before production use."),
]
print(pd.DataFrame(review, columns=["column", "decision", "reason"]).to_string(index=False))
for col, dec, why in review:
    decide("leakage / fairness review", col, dec, why)

step("3.3  Split by TIME, not at random")
cutoff = pd.Timestamp("2025-07-01")
train, test = df[df.application_date < cutoff], df[df.application_date >= cutoff]
show("Train (before Jul 2025) / test (after)", f"{len(train)} / {len(test)}")
decide("splitting", "application_date", f"train < {cutoff.date()} <= test",
           "The model will score FUTURE applicants; a random split hides drift", "random 80/20",
           f"{len(train)} train / {len(test)} test rows")

step("3.4  Model check: does the cleaned data deliver?")
pipe = Pipeline([("features", LoanFeatureBuilder()),
                 ("prep", build_preprocessor(LOAN_NUMERIC + LoanFeatureBuilder.NEW, LOAN_NOMINAL,
                                             ordinal={"education": EDUCATION_LEVELS}, winsorize=(0.005, 0.995))),
                 ("model", LogisticRegression(max_iter=5000))]).fit(train, train[LOAN_TARGET])
auc_clean = roc_auc_score(test[LOAN_TARGET], pipe.predict_proba(test)[:, 1])
raw_d = raw.drop_duplicates().copy()
raw_d["application_date"] = parse_dates(raw_d.application_date)
for c in LOAN_NUMERIC:
    raw_d[c] = pd.to_numeric(raw_d[c], errors="coerce")
rtr, rte = raw_d[raw_d.application_date < cutoff], raw_d[raw_d.application_date >= cutoff]
pipe_raw = Pipeline([("prep", build_preprocessor(LOAN_NUMERIC, LOAN_NOMINAL, ordinal={"education": EDUCATION_LEVELS})),
                     ("model", LogisticRegression(max_iter=5000))]).fit(rtr, rtr[LOAN_TARGET])
auc_raw = roc_auc_score(rte[LOAN_TARGET], pipe_raw.predict_proba(rte)[:, 1])
show("Test AUC - raw data, coerced, no features", auc_raw)
show("Test AUC - documented preparation", auc_clean)

step("3.5  Before / after summary")
summary = pd.DataFrame({
    "raw": [len(raw), raw.duplicated().sum(), raw.employment_type.nunique(), len(contract.validate(raw)),
            f"{pd.to_numeric(raw.annual_income, errors='coerce').notna().mean():.0%}"],
    "prepared": [len(df), df.duplicated().sum(), df.employment_type.nunique(), len(issues),
                 f"{df.annual_income.notna().mean():.0%}"],
}, index=["rows", "exact duplicates", "employment_type categories", "contract violations",
          "usable income values"])
print(summary.to_string())

step("3.6  Columns kept as-is still get ONE line (silence is not documentation)")


def covered_columns(entries):
    return {c.strip() for e in entries for c in e["columns"].split(",")}


untouched = [c for c in raw.columns if c not in covered_columns(log.entries) and c != LOAN_TARGET]
for c in untouched:
    s = df[c]
    ev = (f"range {s.min():g}..{s.max():g}, {s.isna().mean():.1%} missing" if pd.api.types.is_numeric_dtype(s)
          else f"{s.nunique()} categories, {s.isna().mean():.1%} missing: {sorted(s.dropna().unique())[:5]}")
    decide("reviewed - kept as-is", c, "no change", "Profiled and validated: values plausible, consistent",
           evidence=ev)
decide("target", LOAN_TARGET, "kept as-is (0/1)", "Defined by collections: 90+ days past due within 12 months",
           evidence=f"default rate {df[LOAN_TARGET].mean():.1%}")

step("3.7  Export the deliverables")
df.to_csv(OUT_DIR / "12_loans_prepared.csv", index=False)
md = log.to_markdown(OUT_DIR / "12_loan_preprocessing_decision_log.md",
                     header=f"Raw rows: {len(raw):,} (fingerprint {fingerprint(raw)}) -> prepared rows: {len(df):,} "
                            f"(fingerprint {fingerprint(df)}). Time-split test AUC: {auc_clean:.3f} "
                            f"(raw baseline {auc_raw:.3f}).")
log.to_json(OUT_DIR / "12_loan_preprocessing_decision_log.json")
card = [
    "# Data Card: loan applications (prepared v1)", "",
    f"- Source: raw production extract, {len(raw):,} rows, fingerprint `{fingerprint(raw)}`",
    f"- Prepared: {len(df):,} rows, fingerprint `{fingerprint(df)}`",
    f"- Period: {df.application_date.min().date()} to {df.application_date.max().date()}",
    f"- Target: `{LOAN_TARGET}` (rate {df[LOAN_TARGET].mean():.1%})",
    f"- Split: time-based at {cutoff.date()} ({len(train):,} train / {len(test):,} test)",
    f"- Decisions logged: {len(log.entries)} (see 12_loan_preprocessing_decision_log.md)",
    "- Known limitations: income missing-not-at-random; interest_rate reflects the previous model; "
    "age requires legal review; employer excluded in v1.", "",
    "## Schema", "", "| column | dtype | missing % | example |", "|---|---|---|---|",
] + [f"| {c} | {df[c].dtype} | {100 * df[c].isna().mean():.1f} | {str(df[c].dropna().iloc[0])[:30]} |"
     for c in df.columns]
(OUT_DIR / "12_loans_data_card.md").write_text("\n".join(card), encoding="utf-8")
for f in ["12_loans_prepared.csv", md.name, "12_loan_preprocessing_decision_log.json", "12_loans_data_card.md"]:
    print(f"  [saved] outputs/{f}")
show("Decisions documented", len(log.entries))

step("3.8  Self-review against the rubric (the same checks a reviewer will run)")
covered = covered_columns(log.entries)
data_changing = [e for e in log.entries if e["step"] in ("uniqueness", "parsing", "validity")]
checks = {
    "Every raw column has at least one logged decision": all(c in covered for c in raw.columns),
    "Every entry has a rationale (WHY)": all(e["rationale"] for e in log.entries),
    "Every data-changing step records rows affected": all(e["rows_affected"] is not None for e in data_changing),
    "Alternatives recorded for data-changing steps": all(e["alternatives_considered"] for e in data_changing),
    "Data contract passes on prepared data": len(issues) == 0,
    "Leakage / fairness review recorded": any(e["step"].startswith("leakage") for e in log.entries),
    "Split strategy recorded": any(e["step"] == "splitting" for e in log.entries),
    "Prepared data beats the raw baseline": auc_clean > auc_raw,
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
note("""These are the automated rubric checks in ASSIGNMENT.md. A human reviewer
then reads the WHY column: is the evidence a number? Are rejected alternatives
plausible? Would a new team member reproduce the same dataset from the log?""")

banner("KEY TAKEAWAYS")
print("""  1. Profile first; every cleaning step measures what it changed (rows affected, before/after).
  2. A decision = what + which columns + how many rows + WHY (evidence) + alternatives rejected.
  3. Validate the result with the same contract that flagged the raw data.
  4. Record leakage and fairness reviews as decisions too - they are the ones auditors ask about.
  5. Ship: prepared data + decision log + data card + reproducible pipeline + fingerprints.""")
