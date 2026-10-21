"""
LAB 10 - HANDS-ON LAB: A Reusable Preprocessing Pipeline on Real Data (Ames Housing)
=====================================================================================

DATASET
    1,460 real house sales in Ames, Iowa (2006-2010), 79 raw features, De Cock (2011).
    Famous for being REALISTICALLY messy: 19 columns with missing values, most of
    which do not mean "unknown" at all. Target: SalePrice.

TOOLS (as required)
    pandas | SimpleImputer | OneHotEncoder | StandardScaler | ColumnTransformer | Pipeline
    (+ OrdinalEncoder, FunctionTransformer, TransformedTargetRegressor)

WHAT YOU WILL DO
    PART 1 (EASY)   Audit the raw data with pandas - and catch a CSV-parsing bug
    PART 2 (EASY)   Column roles + missingness SEMANTICS ("NA" = no pool != unknown)
    PART 3 (MEDIUM) Evidence-based decisions: skew, log target, outliers, domain features
    PART 4 (MEDIUM) Build the pipeline: ColumnTransformer + Pipeline, train, evaluate
    PART 5 (HARD)   Make it reusable: a config-driven factory, an ABLATION STUDY that
                    proves each decision, model swap, serve a partial raw record,
                    save the artifact + an auto-generated decision log
"""
# %%
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from ml_lab_utils import (HAS_MPL, OUT_DIR, RANDOM_STATE, banner, load_ames, note, plt, save_fig, show,
                          step)
from preprocessing import DecisionLog, fingerprint

warnings.filterwarnings("ignore", message="Found unknown categories")
log = DecisionLog("Ames Housing (1,460 sales) -> SalePrice")

# %% =========================================================================
# PART 1 (EASY): Audit the raw data with pandas
# ============================================================================
banner("PART 1 - Audit the raw data", "EASY")

step("1.1  Load it twice: pandas defaults vs explicit parsing")
naive = load_ames(naive_parsing=True)
df = load_ames()
show("MasVnrType missing - pandas defaults", int(naive.MasVnrType.isna().sum()))
show("MasVnrType missing - only empty cells = NaN", int(df.MasVnrType.isna().sum()))
print(df.MasVnrType.value_counts(dropna=False).to_string())
note("""pandas treats the STRING 'None' as missing by default (also 'NA', 'null',
'n/a' ...). Here 'None' is a real category: 'this house has no masonry veneer'.
Default parsing created 864 fake missing values before we wrote a single line
of ML. Your CSV reader is part of your preprocessing.""")
log.record("parsing", "all", "read_csv(keep_default_na=False, na_values=[''])",
           "The string 'None' is a legitimate category in this data dictionary",
           "pandas defaults", "MasVnrType: 872 'missing' with defaults vs 8 truly empty", 864)

step("1.2  Shape, dtypes, duplicates, target")
show("Rows x columns", df.shape)
print(df.dtypes.value_counts().to_string())
show("Duplicate rows / duplicate Ids", f"{df.duplicated().sum()} / {df.Id.duplicated().sum()}")
print(df.SalePrice.describe().apply(lambda v: f"{v:,.0f}").to_string())
show("Skewness of SalePrice", float(skew(df.SalePrice)))

step("1.3  Missing values")
miss = df.isna().sum()
miss = (miss[miss > 0] / len(df) * 100).sort_values(ascending=False).round(1)
print(miss.rename("% missing").to_string())

step("1.4  Seal a test set NOW (20%), before any decision is made")
train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
show("Train / test rows", f"{len(train)} / {len(test)}")
note("""Every statistic below (skewness, outliers, MI) is computed on TRAIN only.
The test set is opened once, at the very end.""")

# %% =========================================================================
# PART 2 (EASY): Column roles and missingness semantics
# ============================================================================
banner("PART 2 - Column roles and what 'missing' means", "EASY")
QUAL = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
ORDINAL = {
    "ExterQual": QUAL, "ExterCond": QUAL, "BsmtQual": QUAL, "BsmtCond": QUAL, "HeatingQC": QUAL,
    "KitchenQual": QUAL, "FireplaceQu": QUAL, "GarageQual": QUAL, "GarageCond": QUAL, "PoolQC": QUAL,
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"], "PavedDrive": ["N", "P", "Y"], "CentralAir": ["N", "Y"],
}
STRUCTURAL_NONE = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                   "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC",
                   "Fence", "MiscFeature"]
DROP = ["Id", "Utilities", "GarageYrBlt"]
NUM_AS_CATEGORY = ["MSSubClass", "MoSold"]

step("2.1  Structural vs genuine missingness - prove it with the data")
checks = pd.DataFrame({
    "missing": [train.PoolQC.isna().sum(), train.FireplaceQu.isna().sum(), train.GarageType.isna().sum(),
                train.LotFrontage.isna().sum()],
    "explained by": ["PoolArea == 0", "Fireplaces == 0", "GarageArea == 0", "(nothing)"],
    "explained rows": [(train.PoolQC.isna() & (train.PoolArea == 0)).sum(),
                       (train.FireplaceQu.isna() & (train.Fireplaces == 0)).sum(),
                       (train.GarageType.isna() & (train.GarageArea == 0)).sum(), 0],
}, index=["PoolQC", "FireplaceQu", "GarageType", "LotFrontage"])
print(checks.to_string())
note("""PoolQC is missing exactly when the house has no pool: the value is not
unknown, it is 'None'. Imputing the MODE would give over 1,100 houses an
imaginary pool. LotFrontage has no such explanation: it is genuinely unknown ->
impute (median) and keep a missing indicator.""")
log.record("missing values", STRUCTURAL_NONE, "SimpleImputer(strategy='constant', fill_value='None')",
           "Missing means 'feature absent' (no pool / garage / basement / fireplace), proven by the matching "
           "area/count column being 0", "mode imputation (invents pools)",
           f"PoolQC missing {checks.loc['PoolQC', 'missing']} = PoolArea==0 {checks.loc['PoolQC', 'explained rows']}")
log.record("missing values", "LotFrontage", "median imputation + missing indicator",
           "Genuinely unknown (no structural explanation); skewed -> median", "mean; neighbourhood median",
           f"{train.LotFrontage.isna().mean():.1%} missing in train")
log.record("missing values", "MasVnrArea", "fill 0 (with MasVnrType 'None')",
           "8 rows lack both veneer type and area; 'no veneer' is the most common state", "median")

step("2.2  Numbers that are really categories, and columns to drop")
print(pd.DataFrame([
    ["MSSubClass", "int codes 20..190", "dwelling TYPE code -> nominal (one-hot)"],
    ["MoSold", "1..12", "month -> nominal (seasonality, no linear trend)"],
    ["OverallQual", "1..10", "ordered rating -> keep numeric"],
    ["Id", "row id", "DROP - identifier"],
    ["Utilities", "1,459 x 'AllPub'", "DROP - near-constant"],
    ["GarageYrBlt", "year, NaN = no garage", "DROP - r~0.83 with YearBuilt, NaN not fillable"],
], columns=["column", "looks like", "decision"]).to_string(index=False))
log.record("feature types", NUM_AS_CATEGORY, "cast to string -> one-hot", "Integer CODES, not quantities",
           "treat as numeric (fake order)")
log.record("feature selection", DROP, "drop", "identifier / near-constant / redundant with YearBuilt")
log.record("encoding", list(ORDINAL), "OrdinalEncoder with explicit order (None < Po < Fa < TA < Gd < Ex)",
           "Quality scales are ORDERED; one-hot would discard the order and add ~60 columns",
           "one-hot (tested in ablation)")

# %% =========================================================================
# PART 3 (MEDIUM): Evidence-based decisions
# ============================================================================
banner("PART 3 - Decisions backed by evidence", "MEDIUM")

step("3.1  Target: skewed -> model log(SalePrice)")
show("Skew of SalePrice (train)", float(skew(train.SalePrice)))
show("Skew of log1p(SalePrice)", float(skew(np.log1p(train.SalePrice))))
note("""Prices are multiplicative: a better kitchen adds ~X% not ~$X. On the log
scale errors become relative (RMSLE, the competition metric) and a linear model
fits much better. TransformedTargetRegressor applies log on fit and exp on
predict, so callers still get dollars back.""")
log.record("target transform", "SalePrice", "log1p via TransformedTargetRegressor (inverse expm1)",
           "Right-skewed, multiplicative effects; evaluation metric is RMSLE", "raw target",
           f"skew {skew(train.SalePrice):.2f} -> {skew(np.log1p(train.SalePrice)):.2f}")

step("3.2  Outliers: two huge houses sold for very little")
is_outlier = lambda d: (d.GrLivArea > 4000) & (d.SalePrice < 300_000)   # noqa: E731
print(train[train.GrLivArea > 4000][["GrLivArea", "SalePrice", "SaleCondition"]].to_string())
note("""De Cock (the dataset's author) flags these: PARTIAL sales of unfinished
houses - a different process from a normal resale. The big house that sold for
$745k is genuine and stays. This is really a SCOPE decision: our product prices
normal resales of finished homes. Partial sales of 4,000+ sq ft new builds are
out of scope -> removed from training AND evaluation, and the serving layer
must refuse/flag such inputs instead of pricing them (Part 5).""")
log.record("outliers / scope", ["GrLivArea", "SalePrice"],
           "exclude GrLivArea>4000 & SalePrice<300k from training and evaluation; flag such inputs at serving",
           "Partial sales of unfinished homes (documented by the dataset author) - out of product scope",
           "keep; winsorize; drop all >4000 sq ft", f"{is_outlier(train).sum()} training rows")

step("3.3  Skewed numeric features -> log1p (decided on TRAIN)")
AREA_LIKE = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
             "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",
             "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal"]
sk = train[AREA_LIKE].skew().sort_values(ascending=False)
SKEWED = sk[sk.abs() > 0.75].index.tolist() + ["TotalSF", "TotalPorchSF"]
print(sk.round(2).rename("skew (train)").to_string())
log.record("scaling", SKEWED, "log1p, then StandardScaler", "|skew| > 0.75 on train; areas are multiplicative",
           "StandardScaler only; PowerTransformer", f"{len(SKEWED)} columns")

step("3.4  Domain features: what an appraiser looks at")


def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["TotalSF"] = X.TotalBsmtSF.fillna(0) + X["1stFlrSF"] + X["2ndFlrSF"]
    X["TotalBath"] = X.FullBath + 0.5 * X.HalfBath + X.BsmtFullBath.fillna(0) + 0.5 * X.BsmtHalfBath.fillna(0)
    X["HouseAge"] = X.YrSold - X.YearBuilt
    X["YearsSinceRemod"] = X.YrSold - X.YearRemodAdd
    X["TotalPorchSF"] = X[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].sum(axis=1)
    X["QualxSF"] = X.OverallQual * X.TotalSF
    return X


DOMAIN = ["TotalSF", "TotalBath", "HouseAge", "YearsSinceRemod", "TotalPorchSF"]
tmp = add_domain_features(train)
mi_cols = DOMAIN + ["QualxSF", "GrLivArea", "OverallQual", "YearBuilt", "FullBath", "TotRmsAbvGrd", "LotArea"]
mi = pd.Series(mutual_info_regression(tmp[mi_cols].fillna(0), np.log1p(tmp.SalePrice), random_state=RANDOM_STATE),
               index=mi_cols)
corr = tmp[mi_cols].corrwith(np.log1p(tmp.SalePrice))
print(pd.DataFrame({"MI with log(price)": mi, "corr": corr}).sort_values("MI with log(price)", ascending=False)
      .round(3).to_string())
note("""Buyers pay for TOTAL living space, not for '1st floor' and 'basement'
separately; QualxSF (quality x size) says a big house is worth more PER square
foot when it is well built. Both rank at the top of the MI table. MI is a
one-feature-at-a-time screen, though - Part 5's ablation will judge whether
each one actually helps the MODEL.""")
log.record("feature engineering", DOMAIN, "add TotalSF, TotalBath, HouseAge, YearsSinceRemod, TotalPorchSF",
           "Appraiser logic: total space, total baths, age at sale, outdoor space",
           "let the model infer them", "TotalSF is #2 by mutual information with log(price)")

# %% =========================================================================
# PART 4 (MEDIUM): Build the pipeline
# ============================================================================
banner("PART 4 - ColumnTransformer + Pipeline", "MEDIUM")


def ames_basic_clean(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless, rule-based cleaning that must ALSO run at prediction time."""
    X = X.copy()
    for c in NUM_AS_CATEGORY:
        X[c] = X[c].astype("string").astype(object)
    X["MasVnrType"] = X.MasVnrType.fillna("None")
    X["MasVnrArea"] = X.MasVnrArea.fillna(0)
    return X


def make_ames_pipeline(model=None, *, log_target=True, log_skewed=True, domain_features=True,
                       ordinal_quality=True, min_frequency=10, num_impute="median", quality_interaction=False):
    """Config-driven factory: one function, every experiment, one production artifact."""
    base_num = [c for c in df.select_dtypes("number").columns
                if c not in DROP + NUM_AS_CATEGORY + ["SalePrice"]]
    num = base_num + (DOMAIN if domain_features else []) + (["QualxSF"] if quality_interaction else [])
    skewed = [c for c in SKEWED if c in num] if log_skewed else []
    plain = [c for c in num if c not in skewed]
    ordinal = ORDINAL if ordinal_quality else {}
    nominal = [c for c in df.select_dtypes("object").columns if c not in list(ordinal) + DROP] + NUM_AS_CATEGORY
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

    transformers = [
        ("num", make_pipeline(SimpleImputer(strategy=num_impute, add_indicator=True), StandardScaler()), plain),
        ("nom", make_pipeline(SimpleImputer(strategy="constant", fill_value="None"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                            sparse_output=False)), nominal),
    ]
    if skewed:
        transformers.append(("skewed", make_pipeline(SimpleImputer(strategy="median"), log1p, StandardScaler()),
                             skewed))
    if ordinal:
        transformers.append(("ord", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="None"),
            OrdinalEncoder(categories=list(ordinal.values()), handle_unknown="use_encoded_value", unknown_value=-1),
            StandardScaler()), list(ordinal)))
    regressor = model if model is not None else Ridge(alpha=10)
    if log_target:
        regressor = TransformedTargetRegressor(regressor, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([
        ("clean", FunctionTransformer(ames_basic_clean)),
        ("features", FunctionTransformer(add_domain_features)
         if domain_features or quality_interaction else "passthrough"),
        ("prep", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)),
        ("model", regressor),
    ])


def rmsle(y, p):
    return root_mean_squared_error(np.log1p(y), np.log1p(np.clip(p, 1, None)))


def cv_folds(make, data=train, drop_outliers=True, eval_all=False, folds=5):
    """Per-fold RMSLE. Validation folds contain only IN-SCOPE houses (unless eval_all);
    drop_outliers controls whether the TRAINING folds keep the partial sales."""
    scores = []
    for tr_idx, va_idx in KFold(folds, shuffle=True, random_state=RANDOM_STATE).split(data):
        tr, va = data.iloc[tr_idx], data.iloc[va_idx]
        if not eval_all:
            va = va[~is_outlier(va)]
        if drop_outliers:
            tr = tr[~is_outlier(tr)]
        m = make().fit(tr, tr.SalePrice)
        scores.append(rmsle(va.SalePrice, m.predict(va)))
    return np.array(scores)


def cv_score(make, **kw):
    return float(cv_folds(make, **kw).mean())


step("4.1  Fit the full pipeline and look inside")
pipe = make_ames_pipeline()
fit_data = train[~is_outlier(train)]
pipe.fit(fit_data, fit_data.SalePrice)
prep = pipe.named_steps["prep"]
show("Raw input columns", train.shape[1] - 1)
show("Model input features after preprocessing", len(prep.get_feature_names_out()))
print(pd.Series({name: s.stop - s.start for name, s in prep.output_indices_.items() if name != "remainder"},
                name="output columns per branch").to_string())
print(pipe)

step("4.2  Cross-validated performance (RMSLE ~ typical % error)")
baseline = rmsle(train.SalePrice, np.full(len(train), train.SalePrice.median()))
show("Baseline: predict the median price", baseline)
show("Pipeline + Ridge, 5-fold CV RMSLE", cv_score(make_ames_pipeline))

# %% =========================================================================
# PART 5 (HARD): Reusable, proven, served
# ============================================================================
banner("PART 5 - Ablation study, model swap, serving, artifact", "HARD")

step("5.1  Ablation: switch off ONE decision at a time")
configs = {
    "Full pipeline (all decisions)": dict(),
    "- log target": dict(log_target=False),
    "- log1p skewed features": dict(log_skewed=False),
    "- domain features": dict(domain_features=False),
    "- ordinal quality (one-hot instead)": dict(ordinal_quality=False),
    "- min_frequency (all rare levels kept)": dict(min_frequency=None),
    "mean instead of median imputation": dict(num_impute="mean"),
    "+ QualxSF interaction (MI's #1 feature)": dict(quality_interaction=True),
}
fold_scores = {name: cv_folds(lambda kw=kw: make_ames_pipeline(**kw)) for name, kw in configs.items()}
fold_scores["- outlier removal (train on partial sales too)"] = cv_folds(make_ames_pipeline, drop_outliers=False)
full = fold_scores["Full pipeline (all decisions)"]
rows = []
for name, s in fold_scores.items():
    diff = s - full                                   # PAIRED: same folds, same houses
    is_full = name == "Full pipeline (all decisions)"
    se = np.nan if is_full else diff.std(ddof=1) / np.sqrt(len(diff))
    worse, better = int((diff > 0).sum()), int((diff < 0).sum())
    if is_full:
        verdict = ""
    elif diff.mean() > 2 * se or worse == len(diff):
        verdict = "variant WORSE -> decision earns its place"
    elif diff.mean() < -2 * se or better == len(diff):
        verdict = "variant BETTER -> REVISIT the decision"
    else:
        verdict = "within noise -> keep the simpler option"
    rows.append({"variant": name, "CV RMSLE": s.mean(), "change": diff.mean(), "+/- 2 SE": 2 * se,
                 "folds worse": "" if is_full else f"{worse}/{len(diff)}", "verdict": verdict})
abl = pd.DataFrame(rows).set_index("variant")
print(abl.to_string(formatters={"CV RMSLE": "{:.4f}".format, "change": "{:+.4f}".format,
                                "+/- 2 SE": lambda v: "" if np.isnan(v) else f"{v:.4f}"}))
show("Fold-to-fold std of the full pipeline", float(full.std(ddof=1)))
note("""Each variant is compared on the SAME 5 folds (paired). Evidence = the change
exceeds its noise band (+/- 2 SE) or is consistent in every fold.
 - Without the log target, one fold blows up (negative price predictions), so
   the SE is huge - yet it is worse in every fold: the decision clearly stands.
 - Some verdicts say REVISIT: the data disagrees with a v1 design choice. That
   is the system working - the evidence, not the author's taste, decides. The
   next iteration adopts them (try it: it is a one-argument change).
 - 'Within noise' means: keep whichever option is simpler to explain.
This table is the strongest answer to 'why did you preprocess it that way?'""", keep_lines=True)

step("5.1b  Scope changes the verdict: QualxSF when giant houses ARE evaluated")
for name, kw in [("Full pipeline", {}), ("+ QualxSF", {"quality_interaction": True})]:
    show(f"{name}, validation incl. partial sales", cv_score(lambda kw=kw: make_ames_pipeline(**kw), eval_all=True))
note("""A product feature like quality x size EXTRAPOLATES badly: for a 5,600 sq ft
house it produces a value far beyond anything seen in training, and the linear
model multiplies it through. Within scope it is harmless; outside scope it is
a liability. Feature decisions and scope decisions are linked.""")
for name, row in abl.iloc[1:].iterrows():
    log.record("ablation evidence", name.strip("-+ "), row["verdict"],
               "5-fold paired CV: RMSLE of the variant minus the full pipeline",
               evidence=f"{row['change']:+.4f} RMSLE (noise band +/- {row['+/- 2 SE']:.4f})")

step("5.2  Same preprocessing, different model")
models = {"Ridge (alpha=10)": Ridge(alpha=10),
          "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                                                random_state=RANDOM_STATE)}
for name, mdl in models.items():
    show(f"{name}: CV RMSLE", cv_score(lambda mdl=mdl: make_ames_pipeline(mdl)))
note("""The preprocessing factory is model-agnostic: swapping the estimator is one
argument. Here the well-preprocessed LINEAR model beats boosting - the log
transforms and domain features did the heavy lifting - and it is far easier
to explain to a homeowner or an auditor.""")

step("5.3  Open the sealed test set - once")
in_scope = test[~is_outlier(test)]
pred = pipe.predict(in_scope)
show("Test houses in scope / total", f"{len(in_scope)} / {len(test)}")
show("Test RMSLE", rmsle(in_scope.SalePrice, pred))
show("Test MAE", f"${mean_absolute_error(in_scope.SalePrice, pred):,.0f}")
show("Median absolute % error", f"{np.median(np.abs(pred / in_scope.SalePrice - 1)):.1%}")

step("5.4  Serve a PARTIAL raw record (a listing form with most fields empty)")
listing = {"Neighborhood": "CollgCr", "GrLivArea": 1850, "TotalBsmtSF": 900, "1stFlrSF": 950, "2ndFlrSF": 900,
           "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2004, "YearRemodAdd": 2004, "YrSold": 2010,
           "FullBath": 2, "HalfBath": 1, "GarageCars": 2, "GarageArea": 480, "KitchenQual": "Gd",
           "MSSubClass": 60, "MoSold": 6, "SaleCondition": "Normal", "BsmtFullBath": 0, "BsmtHalfBath": 0,
           "WoodDeckSF": 120, "OpenPorchSF": 40}
INPUT_COLUMNS = [c for c in df.columns if c != "SalePrice"]
naive_row = pd.DataFrame([listing]).reindex(columns=INPUT_COLUMNS)    # unprovided fields -> NaN
show("Fields provided / expected", f"{len(listing)} / {len(INPUT_COLUMNS)}")
show("Predicted price - unprovided fields left as NaN", f"${pipe.predict(naive_row)[0]:,.0f}")
note("""Far too low for a 2004, 1,850 sq ft, quality-7 house. Why? In TRAINING data
a missing BsmtQual / GarageQual / ExterQual means 'no basement / no garage /
worst quality' -> our pipeline fills 'None'. In a WEB FORM, missing means
'the user did not fill it in'. Same NaN, different meaning: a serving-time
version of the semantics problem from Part 2.""")
fit_rows = train[~is_outlier(train)]
DEFAULTS = {c: (fit_rows[c].mode().iloc[0] if fit_rows[c].dtype == object else fit_rows[c].median())
            for c in INPUT_COLUMNS}                                   # 'typical house', from TRAIN only
row = pd.DataFrame([{**DEFAULTS, **listing}])[INPUT_COLUMNS]
show("Predicted price - unprovided fields = typical values", f"${pipe.predict(row)[0]:,.0f}")
show("Same house in 'Tesla Town' (unseen neighbourhood)",
     f"${pipe.predict(row.assign(Neighborhood='Tesla Town'))[0]:,.0f}")


def scope_guard(record: dict) -> str | None:
    """Refuse inputs the model was never meant to price."""
    if record.get("GrLivArea", 0) > 4000:
        return "out of scope: > 4,000 sq ft (route to a human appraiser)"
    return None


show("Scope guard for a 5,000 sq ft house", scope_guard({**listing, "GrLivArea": 5000}))
log.record("serving", "all input fields", "fill unprovided fields with TRAIN mode/median before the pipeline; "
           "refuse GrLivArea > 4000", "In a form, missing = 'not provided', not 'feature absent'",
           "leave NaN (structural 'None' -> severely underprices)",
           f"listing priced ${pipe.predict(naive_row)[0]:,.0f} with NaNs vs ${pipe.predict(row)[0]:,.0f} with defaults")

step("5.5  Save the artifact and the decision log")
joblib.dump({"pipeline": pipe, "input_columns": INPUT_COLUMNS, "serving_defaults": DEFAULTS,
             "train_fingerprint": fingerprint(train)}, OUT_DIR / "ames_price_pipeline_v1.joblib")
md = log.to_markdown(OUT_DIR / "10_ames_preprocessing_decisions.md",
                     header=f"Training rows: {len(train)} (test sealed: {len(test)}). "
                            f"Final in-scope test RMSLE: {rmsle(in_scope.SalePrice, pred):.4f}.")
print(f"  [saved] outputs/ames_price_pipeline_v1.joblib")
print(f"  [saved] outputs/{md.name}  ({len(log.entries)} documented decisions)")
print(log.to_frame()[["step", "columns", "decision"]].head(8).to_string(index=False, max_colwidth=60))

if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].hist(train.SalePrice / 1000, bins=50, color="#4c78a8")
    axes[0, 0].set_title(f"SalePrice (skew {skew(train.SalePrice):.2f})")
    axes[0, 0].set_xlabel("$ thousands")
    axes[0, 1].hist(np.log1p(train.SalePrice), bins=50, color="#59a14f")
    axes[0, 1].set_title(f"log1p(SalePrice) (skew {skew(np.log1p(train.SalePrice)):.2f})")
    o = is_outlier(train)
    axes[1, 0].scatter(train.GrLivArea[~o], train.SalePrice[~o] / 1000, s=6, alpha=0.5)
    axes[1, 0].scatter(train.GrLivArea[o], train.SalePrice[o] / 1000, s=40, color="#e15759",
                       label="partial sales (out of scope)")
    axes[1, 0].set_xlabel("GrLivArea (sq ft)")
    axes[1, 0].set_ylabel("SalePrice ($ thousands)")
    axes[1, 0].legend(fontsize=8)
    ab = abl.iloc[1:].sort_values("change")
    colors = {"variant WORSE": "#e15759", "variant BETTER": "#4c78a8"}
    axes[1, 1].barh(ab.index, ab["change"], xerr=ab["+/- 2 SE"],
                    color=[next((c for k, c in colors.items() if v.startswith(k)), "#bab0ac") for v in ab.verdict])
    axes[1, 1].axvline(0, color="black", lw=0.8)
    axes[1, 1].set_xscale("symlog", linthresh=0.005)
    axes[1, 1].set_xticks([-0.01, 0, 0.01, 0.1], ["-0.01", "0", "+0.01", "+0.1"])
    axes[1, 1].set_xlabel("change in RMSLE (red: decision earns its place, blue: revisit)", fontsize=8)
    axes[1, 1].set_title("Ablation: RMSLE change per variant (+/- 2 SE)", fontsize=9)
    axes[1, 1].tick_params(axis="y", labelsize=8)
    save_fig(fig, "10_ames_pipeline")

banner("KEY TAKEAWAYS")
print("""  1. Parsing is preprocessing: 'None' and 'NA' can be real categories.
  2. Ask WHY a value is missing - structural 'None' vs genuinely unknown need different fixes.
  3. Make each decision with evidence from TRAIN data, and log it.
  4. One config-driven pipeline factory -> every experiment, every model, one production artifact.
  5. An ablation table proves which decisions matter; the decision log explains them.""")
