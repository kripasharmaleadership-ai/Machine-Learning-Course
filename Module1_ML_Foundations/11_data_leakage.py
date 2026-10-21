"""
LAB 11 - Data Leakage
=====================

DEFINITION
    Data leakage = information is available during TRAINING (or evaluation) that
    will NOT be available when the model makes real predictions.
    Result: amazing offline scores, disappointing (or disastrous) production.

    "If your model looks too good to be true, it is leaking." - every senior ML engineer

THE MAIN TYPES
    1. TARGET LEAKAGE        : a feature is created AFTER / BECAUSE OF the outcome
                               (cancellation_reason predicts churn... because only churners have one)
    2. TRAIN-TEST CONTAMINATION: preprocessing / feature selection / encoding fitted on ALL data
                               before splitting, so the test set influenced training
    3. TEMPORAL LEAKAGE      : using future information to predict the past (Labs 07, 09)
    4. GROUP / DUPLICATE LEAKAGE: the same entity or row appears in train and test (Lab 07)

LEVELS
    EASY   : Target leakage - leaky churn columns give a fake 0.99+ AUC.
    MEDIUM : Contamination - feature selection and target encoding done BEFORE the split
             find "signal" in pure random noise. The fix: Pipelines.
    HARD   : Build an automated LEAKAGE AUDIT that scans a dataset and flags suspects
             (single-feature power, missingness patterns, duplicates, name heuristics).
"""
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from ml_lab_utils import (RANDOM_STATE, TELCO_CATEGORICAL, TELCO_LEAKY_COLUMNS, TELCO_NUMERIC,
                          TELCO_TARGET, banner, make_telco_churn, note, show, step)

rng = np.random.default_rng(RANDOM_STATE)
df = make_telco_churn()

# %% =========================================================================
# LEVEL 1 (EASY): Target leakage
# ============================================================================
banner("Target leakage: features that 'know' the answer", "EASY")
leaky_df = df.copy()
leaky_df["has_cancellation_reason"] = leaky_df.cancellation_reason.notna().astype(int)
LEAKY_NUM = ["days_since_last_payment", "retention_offer_made", "has_cancellation_reason"]


def churn_pipeline(num_cols):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), TELCO_CATEGORICAL),
        ])),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ])


tr, te = train_test_split(leaky_df, test_size=0.25, stratify=leaky_df[TELCO_TARGET], random_state=RANDOM_STATE)
leaky_model = churn_pipeline(TELCO_NUMERIC + LEAKY_NUM).fit(tr, tr[TELCO_TARGET])
clean_model = churn_pipeline(TELCO_NUMERIC).fit(tr, tr[TELCO_TARGET])
show("WITH leaky columns - test AUC", roc_auc_score(te[TELCO_TARGET], leaky_model.predict_proba(te)[:, 1]))
show("WITHOUT leaky columns - test AUC", roc_auc_score(te[TELCO_TARGET], clean_model.predict_proba(te)[:, 1]))

step("Which features does the leaky model rely on? (permutation importance)")
cols = TELCO_NUMERIC + LEAKY_NUM + TELCO_CATEGORICAL
imp = permutation_importance(leaky_model, te[cols], te[TELCO_TARGET], scoring="roc_auc",
                             n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
print(pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).head(6).round(4).to_string())
note("""The test split did NOT protect us: the leaky columns are in the test set
too, so the test score is just as inflated. On Monday morning, when we score
ACTIVE customers, nobody has a cancellation reason or an overdue payment from
the future -> the model collapses. The question to ask for every column:
'Would I know this value at the moment I need the prediction?'""")

step("What the leaky model does on real Monday-morning data")
monday = te.copy()
monday["days_since_last_payment"] = rng.integers(0, 32, len(monday))   # nobody is overdue yet
monday["retention_offer_made"] = 0
monday["has_cancellation_reason"] = 0
show("Leaky model AUC in production", roc_auc_score(monday[TELCO_TARGET], leaky_model.predict_proba(monday)[:, 1]))
show("Clean model AUC in production", roc_auc_score(monday[TELCO_TARGET], clean_model.predict_proba(monday)[:, 1]))

# %% =========================================================================
# LEVEL 2 (MEDIUM): Train-test contamination
# ============================================================================
banner("Contamination: finding 'signal' in pure noise", "MEDIUM")
note("""Dataset: 200 customers, 5,000 RANDOM features, RANDOM labels. There is
NOTHING to learn - honest accuracy must be ~50%.""")
Xn = rng.normal(size=(200, 5000))
yn = rng.integers(0, 2, 200)
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

step("WRONG: select the 20 'best' features on ALL data, then cross-validate")
selector = SelectKBest(f_classif, k=20).fit(Xn, yn)           # looked at every label, incl. future test folds
X_selected = selector.transform(Xn)
wrong = cross_val_score(LogisticRegression(max_iter=1000), X_selected, yn, cv=cv, scoring="accuracy")
show("Cross-validated accuracy (WRONG)", wrong.mean())

step("RIGHT: feature selection INSIDE the pipeline (refit on each training fold)")
pipe = Pipeline([("select", SelectKBest(f_classif, k=20)), ("clf", LogisticRegression(max_iter=1000))])
right = cross_val_score(pipe, Xn, yn, cv=cv, scoring="accuracy")
show("Cross-validated accuracy (RIGHT)", right.mean())
note("""With 5,000 random columns, some correlate with the labels by pure chance.
Selecting them using ALL labels means the test folds helped pick the features.
This exact mistake has appeared in published medical/genomics papers.
Same rule for scalers, imputers, PCA, encoders, oversampling (SMOTE):
FIT ONLY ON TRAINING DATA - a sklearn Pipeline does it for you automatically.""")

step("Target encoding leakage: a high-cardinality ID-like column")
n = 3000
noise_df = pd.DataFrame({"zip_code": rng.integers(0, 1500, n).astype(str)})  # ~2 rows per zip
y_noise = rng.integers(0, 2, n)
# WRONG: replace each zip by the average label of that zip, computed on ALL rows
zip_mean = pd.Series(y_noise).groupby(noise_df.zip_code).mean()
X_wrong = noise_df.zip_code.map(zip_mean).to_frame()
wrong_auc = cross_val_score(LogisticRegression(), X_wrong, y_noise, cv=cv, scoring="roc_auc").mean()
# RIGHT: sklearn's TargetEncoder uses internal cross-fitting and sits inside the pipeline
right_auc = cross_val_score(make_pipeline(TargetEncoder(cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)),
                                          LogisticRegression()),
                            noise_df, y_noise, cv=cv, scoring="roc_auc").mean()
show("Target-encoded zip, computed on ALL data - AUC", wrong_auc)
show("TargetEncoder inside pipeline - AUC", right_auc)
note("""Each row's own label went into its encoded value - the feature literally
contains the answer. Target/mean encoding is powerful for real categories,
but must be fitted out-of-fold.""")

# %% =========================================================================
# LEVEL 3 (HARD): an automated leakage audit
# ============================================================================
banner("Automated leakage audit - run it on every new dataset", "HARD")

SUSPICIOUS_WORDS = ["cancel", "reason", "after", "outcome", "result", "closed", "refund",
                    "chargeback", "resolution", "final", "status", "retention", "since_last"]


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """How well does THIS ONE column predict the target on held-out data?
    Categorical columns are mean-encoded on one half and scored on the other."""
    half = np.random.default_rng(0).random(len(x)) < 0.5
    if x.dtype == object or str(x.dtype) == "category":
        enc = y[half].groupby(x[half].fillna("NA")).mean()
        s = x[~half].fillna("NA").map(enc).fillna(y[half].mean())
    else:
        s = x[~half].fillna(x[half].median())
    if s.nunique() < 2:
        return 0.5
    auc = roc_auc_score(y[~half], s)
    return max(auc, 1 - auc)


def leakage_audit(data: pd.DataFrame, target: str, id_cols: list[str],
                  test_index: pd.Index | None = None) -> pd.DataFrame:
    y = data[target]
    rows = []
    for col in data.columns.drop([target] + id_cols):
        x = data[col]
        flags = []
        auc = single_feature_auc(x, y)
        if auc > 0.90:
            flags.append(f"single-feature AUC {auc:.2f} (too good)")
        if x.isna().any():
            miss_auc = roc_auc_score(y, x.isna().astype(int))
            if abs(miss_auc - 0.5) > 0.2:
                flags.append(f"missingness predicts target (AUC {max(miss_auc, 1 - miss_auc):.2f})")
        if any(w in col.lower() for w in SUSPICIOUS_WORDS):
            flags.append("suspicious name")
        if x.nunique() / len(x) > 0.95 and x.dtype == object:
            flags.append("near-unique (ID-like)")
        rows.append({"column": col, "single_feature_auc": round(auc, 3),
                     "verdict": "INVESTIGATE" if flags else "ok", "reasons": "; ".join(flags)})
    report = pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False)
    if test_index is not None:
        feats = data.columns.drop([target] + id_cols)
        tr_keys = set(map(tuple, data.drop(index=test_index)[feats].astype(str).to_numpy()))
        dup = sum(tuple(r) in tr_keys for r in data.loc[test_index, feats].astype(str).to_numpy())
        print(f"  Exact duplicate rows shared by train and test: {dup} "
              f"({dup / len(test_index):.1%} of test)")
    return report


# Simulate a messy real-world extract: leaky columns + an accidental duplicate load
messy = df.drop(columns=["signup_date"]).copy()
messy = pd.concat([messy, messy.sample(400, random_state=1)], ignore_index=True)   # ETL loaded a batch twice
_, test_idx = train_test_split(messy.index, test_size=0.25, random_state=RANDOM_STATE)
report = leakage_audit(messy, TELCO_TARGET, id_cols=["customer_id"], test_index=test_idx)
print(report.to_string(index=False))
note("""The audit is a SCREEN, not a judge: 'contract_type' is genuinely
predictive and fine. 'cancellation_reason' is flagged three ways (perfect
missingness pattern, name, power). Duplicates shared across train/test inflate
scores silently - dedupe BEFORE splitting. Wire a check like this into your
training pipeline so a human must sign off on every flagged column.""")

banner("KEY TAKEAWAYS")
print("""  1. For every feature ask: 'Is this value known at prediction time, with this exact definition?'
  2. Too-good-to-be-true scores (AUC > 0.95 on a hard business problem) = assume leakage.
  3. Fit EVERY data-dependent step (scaling, selection, encoding, resampling) inside a Pipeline.
  4. Split by time and by entity; deduplicate before splitting.
  5. Automate a leakage audit and make sign-off part of the release process.""")
