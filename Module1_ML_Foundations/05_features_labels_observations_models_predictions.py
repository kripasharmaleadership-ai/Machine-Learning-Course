"""
LAB 05 - Features, Labels, Observations, Models and Predictions
================================================================

THE VOCABULARY (every ML conversation uses these 5 words)

                    features (X)                               label (y)
             +------------------------------------------+   +-----------+
             | size_sqm | bedrooms | age_years | city   |   | price_k   |
    row 1 -> |   85     |    2     |    10     | Pune   |   |   95      |  <- one OBSERVATION
    row 2 -> |  120     |    3     |     2     | Mumbai |   |  240      |     (a.k.a. example,
    ...      |  ...     |   ...    |    ...    | ...    |   |  ...      |      sample, row)
             +------------------------------------------+   +-----------+
                              |                                   |
                              +----------> MODEL.fit(X, y) <------+
                                                |
                          new house features -> MODEL.predict() -> PREDICTION (y_hat)

    OBSERVATION : one thing we learn from (a house, a customer, a transaction)
    FEATURE     : an input property of that thing, known at prediction time
    LABEL       : the answer we want to predict (only known for history)
    MODEL       : the learned function f(X) -> y  (in practice: a file of numbers)
    PREDICTION  : the model's output for a new observation

LEVELS
    EASY   : Build the table by hand and point at every concept.
    MEDIUM : Real churn data: choosing the unit of observation, which columns are
             NOT features, feature engineering, predict vs predict_proba,
             and the model as a saved artifact.
    HARD   : Production reality - a FEATURE CONTRACT that validates inputs, and a
             live demo of training/serving skew silently corrupting predictions.
"""
# %%
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_lab_utils import (
    OUT_DIR, RANDOM_STATE, TELCO_CATEGORICAL, TELCO_LEAKY_COLUMNS, TELCO_NUMERIC,
    TELCO_TARGET, banner, make_telco_churn, note, show, step,
)

# %% =========================================================================
# LEVEL 1 (EASY): Point at every concept
# ============================================================================
banner("The five words of ML, on a tiny house-price table", "EASY")
houses = pd.DataFrame({
    "size_sqm":  [85, 120, 60, 150, 95, 70, 200, 110],
    "bedrooms":  [2, 3, 1, 4, 2, 2, 5, 3],
    "age_years": [10, 2, 25, 5, 15, 30, 1, 8],
    "price_k":   [95, 160, 55, 210, 100, 60, 300, 140],
})
print(houses)

step("OBSERVATIONS = rows")
show("Number of observations", len(houses))
print("  Observation #0 ->", houses.iloc[0].to_dict())

step("FEATURES (X) and LABEL (y)")
X = houses[["size_sqm", "bedrooms", "age_years"]]   # capital X: a TABLE (2-D)
y = houses["price_k"]                               # lower-case y: a COLUMN (1-D)
show("X shape (rows, features)", X.shape)
show("y shape", y.shape)

step("MODEL = learned function. Train it.")
model = LinearRegression().fit(X, y)
print("  Learned formula: price_k =", f"{model.intercept_:.1f}",
      " ".join(f"{c:+.2f}*{n}" for n, c in zip(X.columns, model.coef_)))
note("""The model IS this formula. 'Training' found the numbers. Each coefficient
reads like: +1 sqm adds about that many thousand to the price, holding the
other features fixed. Negative bedrooms? With SIZE held fixed, more bedrooms
means smaller rooms - and with only 8 rows and correlated features
(size vs bedrooms), coefficients are unstable. Never over-read them.""")

step("PREDICTION for a house the model has never seen")
new_house = pd.DataFrame([{"size_sqm": 100, "bedrooms": 3, "age_years": 4}])
show("Predicted price (k)", float(model.predict(new_house)[0]))
note("""Notice: we pass ONLY features for the new house. There is no label - that
is the whole point. Labels exist only for the past.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Real churn data - decisions a data scientist must make
# ============================================================================
banner("Churn data: what is an observation, what is a feature?", "MEDIUM")
df = make_telco_churn()
print(df.head(3).T)

step("Decision 1: the UNIT OF OBSERVATION")
show("Rows", len(df))
show("Unique customers", df.customer_id.nunique())
note("""One row = one customer, as of a snapshot date (30 Jun 2026). The label
'churned' = did they leave in the NEXT 90 days. Getting this definition right
is a business decision, not a coding one (see Lab 09).""")

step("Decision 2: which columns are NOT features?")
column_roles = pd.DataFrame({
    "column": df.columns,
    "role": [
        "ID - identifies, never predicts" if c == "customer_id" else
        "DATE - use to derive features / split by time" if c == "signup_date" else
        "LABEL (target)" if c == TELCO_TARGET else
        "LEAKAGE - known only AFTER churn, never a feature" if c in TELCO_LEAKY_COLUMNS else
        "FEATURE (numeric)" if c in TELCO_NUMERIC else
        "FEATURE (categorical)" for c in df.columns
    ],
})
print(column_roles.to_string(index=False))
note("""customer_id is unique per row: a model can memorize it but it means
nothing for new customers. Leaky columns (cancellation_reason...) are filled
in AFTER the customer has already left - Lab 11 shows how they fake 99%
accuracy.""")

step("Decision 3: FEATURE ENGINEERING - create better inputs from raw ones")
df["avg_monthly_spend"] = df.total_charges / df.tenure_months       # spending pattern
df["is_new_customer"] = (df.tenure_months <= 6).astype(int)         # business knowledge
df["tickets_per_year_tenure"] = df.support_tickets_90d / np.maximum(df.tenure_months / 12, 0.25)
print(df[["tenure_months", "total_charges", "avg_monthly_spend", "is_new_customer",
          "tickets_per_year_tenure"]].head())
note("""Feature engineering = turning domain knowledge into columns. 'New
customers churn more' is something the sales team knows; we make it explicit.""")

numeric = TELCO_NUMERIC + ["avg_monthly_spend", "is_new_customer", "tickets_per_year_tenure"]
X, y = df[numeric + TELCO_CATEGORICAL], df[TELCO_TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y,
                                                    random_state=RANDOM_STATE)
churn_model = Pipeline([
    ("prep", ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), TELCO_CATEGORICAL),
    ])),
    ("clf", LogisticRegression(max_iter=2000)),
])
churn_model.fit(X_train, y_train)
show("Test ROC-AUC", roc_auc_score(y_test, churn_model.predict_proba(X_test)[:, 1]))

step("Decision 4: what kind of PREDICTION do we need?")
sample = X_test.head(5)
out = pd.DataFrame({
    "predict() -> class": churn_model.predict(sample),
    "predict_proba() -> P(churn)": churn_model.predict_proba(sample)[:, 1].round(3),
    "actual": y_test.head(5).to_numpy(),
}, index=sample.index)
print(out)
note("""predict() applies a hidden 0.5 threshold. The business almost always wants
the PROBABILITY so it can rank customers and choose its own threshold based on
cost and capacity (Lab 10). Never let 0.5 be decided by accident.""")

step("Decision 5: the MODEL is an artifact - a file you version and deploy")
path = OUT_DIR / "churn_model_v1.joblib"
joblib.dump(churn_model, path)
reloaded = joblib.load(path)
show("Saved to", f"outputs/{path.name} ({path.stat().st_size / 1024:.1f} KB)")
show("Reloaded model gives same output", bool(np.allclose(
    reloaded.predict_proba(sample), churn_model.predict_proba(sample))))
coefs = pd.Series(churn_model.named_steps["clf"].coef_[0],
                  index=churn_model.named_steps["prep"].get_feature_names_out()).sort_values()
print("\n  The 'knowledge' inside the file (top +/- coefficients):")
print(pd.concat([coefs.head(4), coefs.tail(4)]).round(3).to_string())

# %% =========================================================================
# LEVEL 3 (HARD): Feature contracts and training/serving skew
# ============================================================================
banner("Production: feature contracts & training/serving skew", "HARD")
note("""In production, features arrive from OTHER systems (CRM, billing, mobile
app). If one of them changes a unit, spelling or default value, the model
does NOT crash - it silently returns garbage. Senior teams prevent this with
a FEATURE CONTRACT: a schema checked on every request.""")


@dataclass
class FeatureSpec:
    name: str
    dtype: str                                  # "numeric" or "categorical"
    min_value: float | None = None
    max_value: float | None = None
    allowed: set = field(default_factory=set)
    nullable: bool = False


def build_contract(train_df: pd.DataFrame) -> dict[str, FeatureSpec]:
    """Derive the contract FROM the training data, so it always matches the model."""
    contract = {}
    for col in numeric:
        s = train_df[col]
        contract[col] = FeatureSpec(col, "numeric", float(s.min()), float(s.max()),
                                    nullable=bool(s.isna().any()))
    for col in TELCO_CATEGORICAL:
        contract[col] = FeatureSpec(col, "categorical", allowed=set(train_df[col].unique()))
    return contract


def validate(request: dict, contract: dict[str, FeatureSpec]) -> list[str]:
    errors = []
    for name, spec in contract.items():
        if name not in request:
            errors.append(f"missing feature '{name}'")
            continue
        v = request[name]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if not spec.nullable:
                errors.append(f"'{name}' is null")
        elif spec.dtype == "numeric":
            tolerance = 0.5 * (spec.max_value - spec.min_value)
            if not (spec.min_value - tolerance <= v <= spec.max_value + tolerance):
                errors.append(f"'{name}'={v:g} far outside training range "
                              f"[{spec.min_value:.0f}, {spec.max_value:.0f}]")
        elif v not in spec.allowed:
            errors.append(f"'{name}'='{v}' unseen category, expected one of {sorted(spec.allowed)}")
    return errors


contract = build_contract(X_train)
good_request = X_test.iloc[0].to_dict()
show("Contract features", len(contract))
show("Valid request errors", validate(good_request, contract) or "none")

step("Skew scenario: billing team switches charges to CENTS, CRM renames a contract type")
bad_request = dict(good_request)
bad_request["monthly_charges"] = good_request["monthly_charges"] * 100
bad_request["contract_type"] = "Month to month"          # hyphens dropped by a new CRM
p_good = churn_model.predict_proba(pd.DataFrame([good_request]))[0, 1]
p_bad = churn_model.predict_proba(pd.DataFrame([bad_request]))[0, 1]
show("P(churn) with correct inputs", p_good)
show("P(churn) with skewed inputs", p_bad)
note("""No exception, no warning - just a completely different answer. The
OneHotEncoder 'ignores' the unknown category (all zeros) and the scaled
charge becomes extreme. This is TRAINING/SERVING SKEW, one of the most common
silent failures of production ML.""")

step("The contract catches it BEFORE the model is called")
for err in validate(bad_request, contract):
    print("   REJECTED:", err)


def safe_predict(request: dict) -> dict:
    """What a production prediction endpoint should look like."""
    errors = validate(request, contract)
    if errors:
        return {"status": "rejected", "errors": errors, "model_version": "churn_model_v1"}
    p = float(churn_model.predict_proba(pd.DataFrame([request]))[0, 1])
    return {"status": "ok", "churn_probability": round(p, 4), "model_version": "churn_model_v1"}


print("\n  safe_predict(good) ->", safe_predict(good_request))
print("  safe_predict(bad)  ->", {k: v for k, v in safe_predict(bad_request).items() if k != "errors"})

banner("KEY TAKEAWAYS")
print("""  1. Observation = row, features = inputs known at prediction time, label = answer from history.
  2. IDs, dates and post-event columns are NOT features - decide each column's role explicitly.
  3. Feature engineering turns business knowledge into model inputs.
  4. Prefer probabilities over hard classes; the threshold is a BUSINESS decision.
  5. A model is a versioned artifact; protect it with a feature contract in production.""")
