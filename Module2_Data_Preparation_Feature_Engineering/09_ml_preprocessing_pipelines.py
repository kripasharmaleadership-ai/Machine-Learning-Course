"""
LAB 09 - ML Preprocessing Pipelines
===================================

THE PROBLEM PIPELINES SOLVE
    Preprocessing written as loose notebook cells has three classic failures:
    1. LEAKAGE       statistics (means, scales, encodings) computed on ALL data
    2. SKEW          serving code re-implements the steps slightly differently
    3. FRAGILITY     pd.get_dummies on ONE new row produces different columns

    A scikit-learn Pipeline bundles every data-dependent step WITH the model:
        fit()      -> learns imputation values, scales, categories ON TRAIN ONLY
        predict()  -> applies exactly the same transformations to new data
        joblib     -> one artifact = preprocessing + model, versioned together

BUILDING BLOCKS
    Pipeline / make_pipeline        a chain of steps; the last one may be a model
    ColumnTransformer               different steps for different column groups
    make_column_selector            choose columns by dtype / name pattern
    FunctionTransformer             wrap a stateless function as a step
    custom TransformerMixin class   stateful or domain logic (fit + transform)
    set_output(transform="pandas")  readable DataFrame outputs with feature names

LEVELS
    EASY   : The notebook way breaks on a single new applicant. The pipeline way doesn't.
    MEDIUM : A real pipeline: custom transformers, column groups, feature names,
             and grid-searching PREPROCESSING choices like hyperparameters.
    HARD   : Production-grade: raw-record in, validated, versioned artifact with
             metadata, a test-suite for the pipeline, and latency checks.
"""
# %%
import json
import platform
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import estimator_html_repr

from ml_lab_utils import (EDUCATION_LEVELS, LOAN_NOMINAL, LOAN_NUMERIC, LOAN_TARGET, OUT_DIR, RANDOM_STATE,
                          banner, make_loan_applications, note, show, step)
from preprocessing import (ColumnRule, DataContract, LoanFeatureBuilder, LoanRawCleaner, Winsorizer,
                           build_preprocessor, fingerprint)

# The unscaled variants below (notebook model, 'scale: passthrough' in the grid) do not
# converge - that IS one of the lessons (Lab 05), so we silence the repeated warning.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

loans = make_loan_applications(n=10000, seed=91)
train, test = train_test_split(loans, test_size=0.25, stratify=loans[LOAN_TARGET], random_state=RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): Why notebook-style preprocessing breaks
# ============================================================================
banner("The notebook way vs the pipeline way", "EASY")

step("1.1  Notebook way: fillna + get_dummies + fit")
cols = ["annual_income", "credit_score", "credit_utilization", "employment_type", "loan_purpose"]
nb = loans[cols].copy()                                     # <- ALL rows, incl. future test rows
nb["credit_score"] = nb.credit_score.fillna(nb.credit_score.mean())   # leak: test rows shape the mean
nb = pd.get_dummies(nb, columns=["employment_type", "loan_purpose"])
nb_model = LogisticRegression(max_iter=5000).fit(nb.loc[train.index], train[LOAN_TARGET])
show("Training matrix columns", nb.shape[1])

step("1.2  Monday morning: score ONE new applicant")
new_applicant = pd.DataFrame([{"annual_income": 48000, "credit_score": np.nan, "credit_utilization": 0.91,
                               "employment_type": "Contract", "loan_purpose": "medical"}])
row = pd.get_dummies(new_applicant.fillna({"credit_score": 690}), columns=["employment_type", "loan_purpose"])
show("Columns produced for the new row", row.shape[1])
try:
    nb_model.predict_proba(row)
except ValueError as e:
    show("predict_proba", f"CRASH -> {str(e).splitlines()[0][:70]}")
note("""get_dummies only creates columns for values present IN THIS DATA. One row
= one category per column = different columns. Teams 'fix' this with reindex()
and hard-coded fill values copied into the API - the start of training/serving
skew. Also note the hard-coded 690: someone typed the training mean into the
serving code. When the model is retrained, will they remember to update it?""")

step("1.3  Pipeline way: the same steps, learned on TRAIN, applied everywhere")
pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         ["annual_income", "credit_score", "credit_utilization"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["employment_type", "loan_purpose"]),
    ])),
    ("model", LogisticRegression(max_iter=5000)),
]).fit(train[cols], train[LOAN_TARGET])
show("P(default) for the new applicant", float(pipe.predict_proba(new_applicant)[0, 1]))
show("Median credit score LEARNED from train", pipe.named_steps["prep"].named_transformers_["num"][0].statistics_[1])
note("""Same code path for 7,500 training rows and for 1 live row. The imputation
value and the categories are learned state INSIDE the artifact - nobody copies
numbers into the API.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): A real pipeline + searching over preprocessing choices
# ============================================================================
banner("A real pipeline: domain features, column groups, names, search", "MEDIUM")

step("2.1  Compose: domain features -> column-wise preprocessing -> model")
ENGINEERED = LoanFeatureBuilder.NEW
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                      ("cap", Winsorizer(0.005, 0.995)),
                      ("scale", StandardScaler())]), LOAN_NUMERIC + ENGINEERED),
    ("nom", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False),
     LOAN_NOMINAL),
    ("ord", OrdinalEncoder(categories=[EDUCATION_LEVELS], handle_unknown="use_encoded_value", unknown_value=-1),
     ["education"]),
], remainder="drop", verbose_feature_names_out=False)
model = Pipeline([("features", LoanFeatureBuilder()), ("prep", prep),
                  ("model", LogisticRegression(max_iter=5000))])
model.set_output(transform="pandas")
model.fit(train, train[LOAN_TARGET])
show("Test AUC", roc_auc_score(test[LOAN_TARGET], model.predict_proba(test)[:, 1]))

step("2.2  Inspect what the pipeline produces (named columns, not a mystery matrix)")
Xt = model[:-1].transform(test.head(3))
print(Xt.T.round(3).head(14).to_string())
show("Total model inputs", Xt.shape[1])
coefs = pd.Series(model["model"].coef_[0], index=model[:-1].get_feature_names_out())
print(coefs.abs().sort_values(ascending=False).head(6).round(3).rename("|coefficient|").to_string())

step("2.3  Grid-search PREPROCESSING choices like hyperparameters")
search_pipe = Pipeline([("features", LoanFeatureBuilder()),
                        ("prep", build_preprocessor(LOAN_NUMERIC + ENGINEERED, LOAN_NOMINAL,
                                                    ordinal={"education": EDUCATION_LEVELS})),
                        ("model", LogisticRegression(max_iter=5000))])
grid = {
    "prep__num__impute__strategy": ["mean", "median"],
    "prep__num__impute__add_indicator": [False, True],
    "prep__num__scale": [StandardScaler(), "passthrough"],          # 'passthrough' = skip the step
}
gs = GridSearchCV(search_pipe, grid, scoring="roc_auc", n_jobs=-1,
                  cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)).fit(train, train[LOAN_TARGET])
res = pd.DataFrame(gs.cv_results_)
view = res[["param_prep__num__impute__strategy", "param_prep__num__impute__add_indicator",
            "param_prep__num__scale", "mean_test_score"]].dropna()
view.columns = ["impute", "indicator", "scale", "CV AUC"]
view["scale"] = view.scale.astype(str).str.replace("()", "", regex=False)
print(view.sort_values("CV AUC", ascending=False).round(4).to_string(index=False))
note("""Every candidate re-fits the imputer and scaler INSIDE each CV fold - no
leakage, by construction. Design decisions (indicator yes/no, mean vs median)
become measured, reproducible choices instead of opinions.""")

html_path = OUT_DIR / "09_pipeline_diagram.html"
html_path.write_text(estimator_html_repr(model), encoding="utf-8")
print(f"  [diagram saved] outputs/{html_path.name}  (open in a browser)")

# %% =========================================================================
# LEVEL 3 (HARD): Production-grade pipeline
# ============================================================================
banner("Raw record in -> validated, versioned, tested artifact", "HARD")

step("3.1  Train on RAW data: the cleaner is part of the pipeline")
raw = make_loan_applications(n=10000, seed=92, messy=True).drop_duplicates()
raw_train, raw_test = train_test_split(raw, test_size=0.25, stratify=raw[LOAN_TARGET], random_state=RANDOM_STATE)
production_model = Pipeline([
    ("clean", LoanRawCleaner()),
    ("features", LoanFeatureBuilder()),
    ("prep", build_preprocessor(LOAN_NUMERIC + ENGINEERED, LOAN_NOMINAL, ordinal={"education": EDUCATION_LEVELS},
                                winsorize=(0.005, 0.995), min_frequency=30)),
    ("model", LogisticRegression(max_iter=5000)),
]).fit(raw_train, raw_train[LOAN_TARGET])
auc = roc_auc_score(raw_test[LOAN_TARGET], production_model.predict_proba(raw_test)[:, 1])
show("Test AUC on RAW, messy records", auc)
note("""Same accuracy as the model trained on clean data in 2.1 - the cleaning rules
now travel WITH the model, so the API can send the raw form submission.""")

step("3.2  Input contract - checked BEFORE predicting")
REQUIRED = ["age", "annual_income", "employment_type", "credit_score", "credit_utilization", "loan_amount",
            "loan_term_months", "interest_rate", "num_credit_lines", "delinquencies_2y", "years_employed",
            "home_ownership", "state", "loan_purpose", "education"]
serving_contract = DataContract({
    **{c: ColumnRule("any", max_missing=0.5) for c in REQUIRED},
    "loan_term_months": ColumnRule("number", allowed=[36, 60]),
    "interest_rate": ColumnRule("number", min=0.0, max=0.5),
    "loan_amount": ColumnRule("number", min=500, max=1_000_000),
})
bad_batch = raw_test.head(200).drop(columns=["interest_rate"]).assign(loan_term_months=48)
print(serving_contract.validate(bad_batch).to_string(index=False))
note("""Reject (or route to manual review) BEFORE the model sees it. A model given
a missing column or an unknown loan term will still return a number - a
confident, wrong number.""")

step("3.3  Save ONE artifact + metadata (the model card's technical half)")
artifact = OUT_DIR / "loan_default_pipeline_v1.joblib"
joblib.dump(production_model, artifact)
meta = {
    "model_name": "loan_default_pipeline", "version": "1.0.0",
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python": platform.python_version(), "sklearn": sklearn.__version__,
    "training_rows": len(raw_train), "training_data_fingerprint": fingerprint(raw_train),
    "input_columns": REQUIRED, "model_inputs": list(production_model[:-1].get_feature_names_out()),
    "test_auc": round(auc, 4),
    "steps": [name for name, _ in production_model.steps],
}
(OUT_DIR / "loan_default_pipeline_v1.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"  [saved] outputs/{artifact.name} + .json   ({artifact.stat().st_size / 1024:.0f} KB)")
reloaded = joblib.load(artifact)
same = np.array_equal(reloaded.predict_proba(raw_test), production_model.predict_proba(raw_test))
show("Reloaded artifact gives identical predictions", same)

step("3.4  A test-suite for the pipeline (run it in CI on every change)")
p_batch = production_model.predict_proba(raw_test)[:, 1]
one = raw_test.iloc[[7]]


def t_fit_on_train_only():
    learned = production_model["prep"].named_transformers_["num"]["impute"].statistics_
    expected = production_model[:2].transform(raw_train)[LOAN_NUMERIC + ENGINEERED].median().to_numpy()
    return np.allclose(learned, expected)


def t_single_row_equals_batch():
    return np.isclose(production_model.predict_proba(one)[0, 1], p_batch[7])


def t_column_order_irrelevant():
    shuffled = raw_test[raw_test.columns[::-1]]
    return np.allclose(production_model.predict_proba(shuffled)[:, 1], p_batch)


def t_unseen_category():
    r = one.assign(employment_type="Gig worker", state="ZZ", loan_purpose="crypto")
    return np.isfinite(production_model.predict_proba(r)).all()


def t_missing_optional_fields():
    r = one.assign(credit_score=np.nan, years_employed=np.nan, annual_income="N/A")
    return np.isfinite(production_model.predict_proba(r)).all()


def t_extreme_values_bounded():
    r = one.assign(annual_income="$50,000,000", loan_amount=10_000_000)
    p = production_model.predict_proba(r)[0, 1]
    return 0 <= p <= 1


def t_input_not_mutated():
    before = raw_test.copy()
    production_model.predict_proba(raw_test)
    return before.equals(raw_test)


def t_extra_columns_ignored():
    return np.allclose(production_model.predict_proba(raw_test.assign(utm_campaign="spring"))[:, 1], p_batch)


def t_monotonic_in_credit_score():
    r = pd.concat([one.assign(credit_score=s) for s in [550, 650, 750]])
    p = production_model.predict_proba(r)[:, 1]
    return p[0] > p[1] > p[2]


tests = [t_fit_on_train_only, t_single_row_equals_batch, t_column_order_irrelevant, t_unseen_category,
         t_missing_optional_fields, t_extreme_values_bounded, t_input_not_mutated, t_extra_columns_ignored,
         t_monotonic_in_credit_score]
passed = 0
for t in tests:
    try:
        ok = bool(t())
    except Exception as e:                       # noqa: BLE001 - a test harness reports, never crashes
        ok, _ = False, print(f"     error: {e}")
    passed += ok
    print(f"  {'PASS' if ok else 'FAIL'}  {t.__name__}")
show("Pipeline tests passed", f"{passed}/{len(tests)}")
note("""These are not model-quality tests - they are SOFTWARE tests for data
handling: invariances (order, extra columns), robustness (unknowns, missing,
extremes), statelessness (no mutation) and a BEHAVIOURAL check (higher score
-> lower risk). Behavioural tests catch silent bugs that AUC never shows.""")

step("3.5  Latency budget")
for n_rows in [1, 1000]:
    batch = raw_test.head(n_rows)
    t0 = time.perf_counter()
    for _ in range(20):
        production_model.predict_proba(batch)
    ms = (time.perf_counter() - t0) / 20 * 1000
    show(f"predict_proba, batch of {n_rows}", f"{ms:.1f} ms  ({ms / n_rows * 1000:.0f} us per row)")
note("""Per-call overhead dominates single-row scoring (pandas + ColumnTransformer
bookkeeping). For a real-time API with a 50 ms budget this is fine; for
10k requests/s you would batch requests or export the pipeline (e.g. to ONNX).""")

banner("KEY TAKEAWAYS")
print("""  1. Every data-dependent step goes inside the Pipeline, fitted on train only.
  2. ColumnTransformer routes column groups; custom transformers carry domain logic.
  3. Treat preprocessing choices as hyperparameters: search them with CV.
  4. Ship ONE artifact (clean -> features -> prep -> model) with metadata and a data fingerprint.
  5. Validate inputs with a contract, and test the pipeline like software (invariances, edge cases).""")
