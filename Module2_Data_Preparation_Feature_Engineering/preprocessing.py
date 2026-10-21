"""
preprocessing.py - a small, REUSABLE preprocessing library (built up across Module 2).

In a real ML team, preprocessing is not notebook code. It is a versioned library
with tests, imported by training jobs AND by the serving service, so both run the
exact same transformations (no training/serving skew).

Contents
    1. Raw-value parsing   normalize_category(), parse_money(), parse_dates()
    2. Transformers        Winsorizer, LoanFeatureBuilder   (sklearn-compatible)
    3. Factory             build_preprocessor()  -> ColumnTransformer from a config
    4. Data contract       DataContract          -> validate inputs before training/serving
    5. Governance          DecisionLog, fingerprint()  -> document & version every decision
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   RobustScaler, StandardScaler)
from sklearn.utils.validation import check_is_fitted, validate_data

MISSING_TOKENS = ["", "n/a", "na", "null", "none", "nan", "-", "?", "unknown"]


# ---------------------------------------------------------------------------
# 1. Raw-value parsing (pandas). Run BEFORE the sklearn pipeline, on every source.
# ---------------------------------------------------------------------------
def normalize_category(s: pd.Series, mapping: dict | None = None) -> pd.Series:
    """' SELF_EMPLOYED' -> 'self employed' -> (optional synonym mapping) -> canonical.
    Missing tokens ('N/A', '', 'unknown', ...) become real NaN."""
    out = (s.astype("string").str.strip().str.lower()
           .str.replace(r"[\s_\-]+", " ", regex=True))
    out = out.mask(out.isin(MISSING_TOKENS))
    if mapping:
        out = out.replace(mapping)
    return out.astype(object).where(out.notna(), np.nan)


def parse_money(s: pd.Series) -> pd.Series:
    """'$54,300' / '54300.0' / 'N/A' / '' -> 54300.0 / 54300.0 / NaN / NaN"""
    txt = s.astype("string").str.strip().str.replace(r"[$,\s]", "", regex=True)
    txt = txt.mask(txt.str.lower().isin(MISSING_TOKENS))
    return pd.to_numeric(txt, errors="coerce").astype(float)


def parse_dates(s: pd.Series) -> pd.Series:
    """Accepts several formats in one column ('2025-03-14', '03/14/2025').
    Unparseable values become NaT (and should be counted, not silently ignored)."""
    return pd.to_datetime(s, format="mixed", errors="coerce")


# ---------------------------------------------------------------------------
# 2. Custom transformers
# ---------------------------------------------------------------------------
class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cap each numeric column at quantiles LEARNED ON TRAINING DATA.
    Production values outside the training range are clipped the same way."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float, reset=False)
        return np.clip(X, self.lower_, self.upper_)


class LoanRawCleaner(TransformerMixin, BaseEstimator):
    """Raw API / database record -> typed, consistent columns.
    Stateless, rule-based cleaning (the rules are documented business logic).
    Putting it INSIDE the pipeline means serving accepts the same raw format
    as training - no separate, drifting 'cleaning script' in the web service."""

    EMPLOYMENT = {"salaried": "Salaried", "self employed": "Self-employed", "contract": "Contract",
                  "contractor": "Contract", "unemployed": "Unemployed", "not employed": "Unemployed",
                  "retired": "Retired"}
    HOME = {"rent": "Rent", "mortgage": "Mortgage", "own": "Own", "other": "Other", "none": "Other"}
    NUMERIC = ["age", "years_employed", "credit_score", "num_credit_lines", "credit_utilization",
               "delinquencies_2y", "loan_amount", "loan_term_months", "interest_rate"]

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["annual_income"] = parse_money(out["annual_income"])
        for c in self.NUMERIC:
            if c in out:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out["age"] = out["age"].where(out["age"].between(18, 100))                       # 0 / 999 -> NaN
        out["credit_score"] = out["credit_score"].where(out["credit_score"].between(300, 850))
        util = out["credit_utilization"]
        out["credit_utilization"] = util.where(util <= 1.5, util / 100)                    # % -> fraction
        out["employment_type"] = normalize_category(out["employment_type"], self.EMPLOYMENT)
        out["home_ownership"] = normalize_category(out["home_ownership"], self.HOME)
        out["state"] = out["state"].astype("string").str.strip().str.upper().astype(object)
        out["loan_purpose"] = normalize_category(out["loan_purpose"]).str.replace(" ", "_")
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class LoanFeatureBuilder(TransformerMixin, BaseEstimator):
    """Domain features a credit-risk analyst would compute by hand.
    Stateless (no fit needed) - but wrapping it as a transformer means it is
    saved INSIDE the model artifact and runs identically at serving time."""

    NEW = ["monthly_payment", "payment_to_income", "loan_to_income", "high_utilization",
           "thin_file", "employment_stability"]

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        r = X["interest_rate"] / 12
        n = X["loan_term_months"]
        X["monthly_payment"] = X["loan_amount"] * r / (1 - (1 + r) ** -n)      # annuity formula
        X["payment_to_income"] = X["monthly_payment"] / (X["annual_income"] / 12)
        X["loan_to_income"] = X["loan_amount"] / X["annual_income"]
        X["high_utilization"] = (X["credit_utilization"] > 0.8).astype(float)
        X["thin_file"] = (X["num_credit_lines"] <= 1).astype(float)
        X["employment_stability"] = X["years_employed"] / (X["age"] - 17).clip(lower=1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + self.NEW, dtype=object)


# ---------------------------------------------------------------------------
# 3. Factory: one function, many experiments
# ---------------------------------------------------------------------------
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler, None: None}


def build_preprocessor(numeric: list[str], nominal: list[str] | None = None,
                       ordinal: dict[str, list] | None = None, log_cols: list[str] | None = None,
                       num_impute: str = "median", add_indicator: bool = True,
                       scaler: str | None = "standard", winsorize: tuple | None = None,
                       min_frequency: int | float | None = None) -> ColumnTransformer:
    """
    Build a ColumnTransformer from a CONFIG instead of copy-pasting pipelines.

    numeric   : continuous / count columns            -> impute -> (winsorize) -> scale
    log_cols  : right-skewed positive columns         -> impute -> log1p -> scale
    nominal   : unordered categories                  -> impute 'missing' -> one-hot
    ordinal   : {col: [ordered categories]}           -> impute most_frequent -> integer codes
    """
    nominal, ordinal, log_cols = nominal or [], ordinal or {}, log_cols or []
    plain_num = [c for c in numeric if c not in log_cols]

    def num_steps(extra):
        steps = [("impute", SimpleImputer(strategy=num_impute, add_indicator=add_indicator))]
        steps += extra
        if winsorize:
            steps.append(("winsorize", Winsorizer(*winsorize)))
        if SCALERS[scaler]:
            steps.append(("scale", SCALERS[scaler]()))
        return Pipeline(steps)

    transformers = []
    if plain_num:
        transformers.append(("num", num_steps([]), plain_num))
    if log_cols:
        log1p = FunctionTransformer(np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")
        transformers.append(("log", num_steps([("log1p", log1p)]), log_cols))
    if nominal:
        transformers.append(("nom", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist" if min_frequency else "ignore",
                                     min_frequency=min_frequency, sparse_output=False)),
        ]), nominal))
    if ordinal:
        transformers.append(("ord", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=list(ordinal.values()),
                                       handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), list(ordinal)))
    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# 4. Data contract: catch bad data BEFORE it reaches the model
# ---------------------------------------------------------------------------
@dataclass
class ColumnRule:
    kind: str = "number"                     # "number" | "category" | "any"
    min: float | None = None
    max: float | None = None
    allowed: list | None = None
    max_missing: float = 1.0
    required: bool = True


@dataclass
class DataContract:
    rules: dict[str, ColumnRule]
    unique_key: str | None = None
    max_duplicate_rate: float = 0.0

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return one row per violation (empty frame = data is OK)."""
        issues = []

        def add(col, check, detail, n=0):
            issues.append({"column": col, "check": check, "detail": detail, "rows": int(n)})

        for col, rule in self.rules.items():
            if col not in df:
                if rule.required:
                    add(col, "missing column", "column not found in input")
                continue
            s = df[col]
            miss = s.isna().mean()
            if miss > rule.max_missing:
                add(col, "too many missing", f"{miss:.1%} > {rule.max_missing:.0%}", s.isna().sum())
            if rule.kind == "number":
                num = pd.to_numeric(s, errors="coerce")
                bad_type = num.isna() & s.notna()
                if bad_type.any():
                    add(col, "not numeric", f"e.g. {s[bad_type].iloc[0]!r}", bad_type.sum())
                if rule.min is not None and (num < rule.min).any():
                    add(col, "below min", f"min allowed {rule.min}, got {num.min()}", (num < rule.min).sum())
                if rule.max is not None and (num > rule.max).any():
                    add(col, "above max", f"max allowed {rule.max}, got {num.max()}", (num > rule.max).sum())
            if rule.allowed is not None:
                bad = s.notna() & ~s.isin(rule.allowed)
                if bad.any():
                    add(col, "unexpected category", f"e.g. {sorted(s[bad].astype(str).unique())[:4]}", bad.sum())
        if self.unique_key and self.unique_key in df:
            d = df[self.unique_key].duplicated().sum()
            if d:
                add(self.unique_key, "duplicate key", f"{d} repeated keys", d)
        dup_rate = df.duplicated().mean()
        if dup_rate > self.max_duplicate_rate:
            add("<all>", "duplicate rows", f"{dup_rate:.1%} exact duplicate rows", df.duplicated().sum())
        return pd.DataFrame(issues, columns=["column", "check", "detail", "rows"])


# ---------------------------------------------------------------------------
# 5. Governance: decision log + data fingerprint
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """Short, stable hash of a DataFrame's content -> 'which data trained this model?'"""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:12]


@dataclass
class DecisionLog:
    """Every preprocessing decision, with WHY, the alternatives and the evidence.
    This is what a reviewer, an auditor - or you in 6 months - will ask for."""
    dataset: str
    entries: list = field(default_factory=list)

    def record(self, step: str, columns, decision: str, rationale: str,
               alternatives: str = "", evidence: str = "", rows_affected: int | None = None) -> None:
        cols = columns if isinstance(columns, str) else ", ".join(columns)
        self.entries.append({"#": len(self.entries) + 1, "step": step, "columns": cols,
                             "decision": decision, "rationale": rationale,
                             "alternatives_considered": alternatives, "evidence": evidence,
                             "rows_affected": rows_affected})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)

    def to_markdown(self, path: Path, header: str = "") -> Path:
        lines = [f"# Preprocessing Decision Log: {self.dataset}", "", header, ""]
        for e in self.entries:
            lines += [f"## {e['#']}. [{e['step']}] {e['columns']}",
                      f"- **Decision:** {e['decision']}",
                      f"- **Why:** {e['rationale']}"]
            if e["alternatives_considered"]:
                lines.append(f"- **Alternatives considered:** {e['alternatives_considered']}")
            if e["evidence"]:
                lines.append(f"- **Evidence:** {e['evidence']}")
            if e["rows_affected"] is not None:
                lines.append(f"- **Rows affected:** {e['rows_affected']:,}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def to_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.entries, indent=2, default=str), encoding="utf-8")
        return path
