"""
LAB 14 - HANDS-ON LAB: From Raw Data to a First Baseline Model (Titanic, real data)
====================================================================================

DATASET
    1,309 real passengers of the RMS Titanic (1912), from OpenML.
    Question: "Given what we knew about a passenger AT BOARDING, would they survive?"

WHAT YOU WILL DO (mirrors the first week of a real ML project)
    PART 1  (EASY)   Explore the dataset with pandas
    PART 2  (EASY)   Identify numerical, categorical, text, ID, target - and LEAKY - columns
    PART 3  (MEDIUM) Exploratory data analysis: distributions, survival patterns, missing data
    PART 4  (MEDIUM) Create training and testing datasets (stratified, verified)
    PART 5  (HARD)   Build a ladder of baselines -> a scikit-learn Pipeline model,
                     catch a leakage trap, evaluate honestly, analyse errors,
                     save the model + a metrics report, and predict for new passengers.

RUN
    python 14_hands_on_lab_titanic.py          (needs internet on the FIRST run only)
"""
# %%
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml_lab_utils import (HAS_MPL, OUT_DIR, RANDOM_STATE, banner, load_titanic, note, plt,
                          save_fig, show, step)

# %% =========================================================================
# PART 1 (EASY): Explore the dataset with pandas
# ============================================================================
banner("PART 1 - Explore the dataset with pandas", "EASY")
df = load_titanic()

step("1.1  Shape: how many rows (observations) and columns?")
show("Rows x columns", df.shape)

step("1.2  First rows: ALWAYS look at the raw data with your own eyes")
print(df.head(5).to_string())

step("1.3  Data types: how pandas stores each column")
print(df.dtypes.to_string())
note("""object = text/strings. int64/float64 = numbers. A column stored as a number
is NOT automatically a numerical feature (pclass 1/2/3 is a category with an
order).""")

step("1.4  Summary statistics for numeric columns")
print(df.describe().T.round(2).to_string())
note("""Read it like a detective: age min 0.17 (a baby, 2 months old), fare max
512 vs median ~14 (very skewed), body has very few non-null values - why?""")

step("1.5  Summary for text/categorical columns")
print(df.describe(include="object").T.to_string())

step("1.6  Missing values - count and percentage")
missing = pd.DataFrame({"missing": df.isna().sum(), "pct": (df.isna().mean() * 100).round(1)})
print(missing[missing.missing > 0].sort_values("pct", ascending=False).to_string())

step("1.7  Target distribution")
print(df.survived.value_counts().rename({0: "died (0)", 1: "survived (1)"}).to_string())
show("Survival rate", f"{df.survived.mean():.1%}")
note("""38% survived. So a 'model' that always predicts 'died' is already ~62%
accurate. Remember this number: every real model must beat it.""")

step("1.8  Unique values per column (cardinality)")
print(df.nunique().sort_values(ascending=False).to_string())

# %% =========================================================================
# PART 2 (EASY): Identify numerical, categorical, target and other variable types
# ============================================================================
banner("PART 2 - Identify variable types and roles", "EASY")


def auto_profile(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """First-pass, AUTOMATIC guess of each column's type. A human reviews it next."""
    rows = []
    for col in frame.columns:
        s = frame[col]
        n_unique, ratio = s.nunique(), s.nunique() / len(s)
        if col == target:
            kind = "TARGET"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric-discrete" if n_unique <= 10 else "numeric-continuous"
        elif ratio > 0.5:
            kind = "text / ID-like (high cardinality)"
        else:
            kind = "categorical"
        rows.append({"column": col, "dtype": str(s.dtype), "unique": n_unique,
                     "missing_%": round(s.isna().mean() * 100, 1), "auto_type": kind,
                     "example": s.dropna().iloc[0]})
    return pd.DataFrame(rows)


profile = auto_profile(df, "survived")
print(profile.to_string(index=False))

step("2.1  Human review: the DATA DICTIONARY with role decisions")
data_dictionary = pd.DataFrame([
    ["pclass", "categorical (ordinal)", "FEATURE", "Ticket class 1/2/3 - ordered category, proxy for wealth"],
    ["sex", "categorical (nominal)", "FEATURE", "'Women and children first'"],
    ["age", "numerical (continuous)", "FEATURE", "20% missing -> needs imputation"],
    ["sibsp", "numerical (discrete)", "FEATURE", "# siblings/spouses aboard"],
    ["parch", "numerical (discrete)", "FEATURE", "# parents/children aboard"],
    ["fare", "numerical (continuous)", "FEATURE", "Heavily skewed; 1 missing"],
    ["embarked", "categorical (nominal)", "FEATURE", "Port: C=Cherbourg, Q=Queenstown, S=Southampton"],
    ["name", "text", "FEATURE SOURCE", "Raw name is useless, but the TITLE (Mr/Mrs/Master) is gold"],
    ["cabin", "categorical (sparse)", "FEATURE SOURCE", "77% missing; 'has_cabin' and deck letter are useful"],
    ["ticket", "text / ID-like", "DROP", "Near-unique code; risk of memorizing groups"],
    ["home.dest", "text", "DROP", "Free text, 43% missing, little value for a first model"],
    ["boat", "categorical", "LEAKAGE - DROP", "Lifeboat number: only known IF you survived"],
    ["body", "numerical", "LEAKAGE - DROP", "Body ID number: only known if you DIED and were found"],
    ["survived", "binary", "TARGET", "1 = survived, 0 = died"],
], columns=["column", "variable type", "role", "reason"])
print(data_dictionary.to_string(index=False))
note("""The automatic profiler cannot know that 'boat' and 'body' are recorded AFTER
the sinking. Only domain thinking catches that: 'Would I know this at
boarding time?' Part 5 shows what happens if you keep them.""")

TARGET = "survived"
NUMERIC = ["age", "sibsp", "parch", "fare"]
CATEGORICAL = ["pclass", "sex", "embarked"]
LEAKY = ["boat", "body"]

# %% =========================================================================
# PART 3 (MEDIUM): Exploratory Data Analysis
# ============================================================================
banner("PART 3 - Exploratory data analysis", "MEDIUM")

step("3.1  Survival rate by each categorical feature")
for col in CATEGORICAL:
    t = df.groupby(col, dropna=False)[TARGET].agg(survival_rate="mean", passengers="size")
    print(f"\n  by {col}:\n{t.round(3).to_string()}")
note("""Women survived ~73% vs men ~19%. 1st class ~62% vs 3rd class ~26%.
These two features alone tell most of the story.""")

step("3.2  Interaction: sex x class (a pivot table)")
pivot = df.pivot_table(index="sex", columns="pclass", values=TARGET, aggfunc="mean").round(3)
print(pivot.to_string())
note("""Women in 1st/2nd class: >88% survived. Men in 2nd/3rd class: ~15%. Effects
COMBINE - a good model must capture interactions like this.""")

step("3.3  Numerical features vs survival")
df["age_group"] = pd.cut(df.age, bins=[0, 12, 18, 30, 45, 60, 80],
                         labels=["child", "teen", "young adult", "adult", "middle age", "senior"])
print(df.groupby("age_group", observed=True)[TARGET].agg(["mean", "size"]).round(3).to_string())
df["fare_quartile"] = pd.qcut(df.fare, 4, labels=["Q1 cheap", "Q2", "Q3", "Q4 expensive"])
print(df.groupby("fare_quartile", observed=True)[TARGET].agg(["mean", "size"]).round(3).to_string())
show("Fare skewness (0 = symmetric)", float(df.fare.skew()))

step("3.4  Is MISSINGNESS itself informative?")
for col in ["age", "cabin"]:
    rates = df.groupby(df[col].isna())[TARGET].mean()
    print(f"  {col:<6} known -> survival {rates[False]:.1%} | missing -> survival {rates[True]:.1%}")
note("""Passengers WITHOUT a recorded cabin survived far less - mostly 3rd class.
Missing data is often NOT random; 'has_cabin' becomes a useful feature.""")

step("3.5  Correlation between numeric columns")
print(df[NUMERIC + ["pclass", TARGET]].corr().round(2).to_string())
note("""fare and pclass are strongly (negatively) correlated - both measure wealth.
Correlation with the target shows LINEAR relationships only.""")

step("3.6  Feature ideas from EDA (feature engineering)")


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn EDA insights into features. Pure function: same input -> same output,
    no learned statistics, so it is safe to apply before the split."""
    out = frame.copy()
    title = out["name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    out["title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    out["family_size"] = out["sibsp"] + out["parch"] + 1
    out["is_alone"] = (out["family_size"] == 1).astype(int)
    out["has_cabin"] = out["cabin"].notna().astype(int)
    out["deck"] = out["cabin"].str[0].fillna("Unknown")
    out["log_fare"] = np.log1p(out["fare"])
    return out


fe = add_features(df)
print(fe.groupby("title")[TARGET].agg(["mean", "size"]).round(3).to_string())
print(fe.groupby("family_size")[TARGET].agg(["mean", "size"]).round(3).to_string())
note("""'Master' = boys: high survival, which 'sex' alone would miss. Families of
2-4 did best; solo travellers and very large families did worst (non-linear!).""")

if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    df.groupby("sex")[TARGET].mean().plot.bar(ax=axes[0, 0], title="Survival rate by sex", rot=0)
    pivot.T.plot.bar(ax=axes[0, 1], title="Survival by class and sex", rot=0)
    for s, lbl in [(1, "survived"), (0, "died")]:
        axes[1, 0].hist(df.loc[df[TARGET] == s, "age"].dropna(), bins=30, alpha=0.6, label=lbl)
    axes[1, 0].set(title="Age distribution", xlabel="age")
    axes[1, 0].legend()
    fe.groupby("family_size")[TARGET].mean().plot(ax=axes[1, 1], marker="o", title="Survival by family size")
    save_fig(fig, "14_titanic_eda")

df = df.drop(columns=["age_group", "fare_quartile"])   # EDA helpers only

# %% =========================================================================
# PART 4 (MEDIUM): Create training and testing datasets
# ============================================================================
banner("PART 4 - Create training and testing datasets", "MEDIUM")
FEATURE_SOURCE_COLS = NUMERIC + CATEGORICAL + ["name", "cabin"]
X = df[FEATURE_SOURCE_COLS]          # leaky columns and the target are NOT in X
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

split_check = pd.DataFrame({
    "rows": [len(X_train), len(X_test)],
    "survival_rate": [y_train.mean(), y_test.mean()],
    "female_share": [X_train.sex.eq("female").mean(), X_test.sex.eq("female").mean()],
    "first_class_share": [X_train.pclass.eq(1).mean(), X_test.pclass.eq(1).mean()],
}, index=["train", "test"]).round(3)
print(split_check.to_string())
show("Overlapping passengers (index)", len(set(X_train.index) & set(X_test.index)))
note("""Checklist: (1) target excluded from X, (2) leaky columns excluded,
(3) stratified -> same survival rate in both, (4) key groups similar,
(5) zero overlap. From now on the TEST set is sealed until the final
evaluation. Strictly, EDA should also be done on train only - for a famous
public dataset we relaxed that, in a real project don't.""")

# %% =========================================================================
# PART 5 (HARD): Baselines -> pipeline model -> honest evaluation
# ============================================================================
banner("PART 5 - Build a first baseline model with scikit-learn", "HARD")
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

step("5.1  The baseline ladder (each rung must beat the one below)")


class RuleModel:
    """Wrap a hand-written rule so it can be scored like a model."""
    def __init__(self, rule):
        self.rule = rule

    def predict(self, X_):
        return self.rule(X_).astype(int).to_numpy()


rules = {
    "rung 0: always 'died' (majority)": RuleModel(lambda d: pd.Series(0, index=d.index)),
    "rung 1: women survive": RuleModel(lambda d: d.sex.eq("female")),
    "rung 2: women + 1st/2nd-class children <13 survive":
        RuleModel(lambda d: d.sex.eq("female") | (d.age.lt(13) & d.pclass.lt(3))),
}
ladder = []
for name, m in rules.items():
    pred = m.predict(X_train)
    ladder.append({"model": name, "train_accuracy": accuracy_score(y_train, pred),
                   "train_f1": f1_score(y_train, pred, zero_division=0)})
print(pd.DataFrame(ladder).round(3).to_string(index=False))
note("""A one-line rule ('women survive') is ~78% accurate. That is the bar a
machine-learning model must clear to be worth its complexity.""")

step("5.2  The ML pipeline: feature engineering -> preprocessing -> model")
ENG_NUMERIC = ["age", "sibsp", "parch", "log_fare", "family_size", "is_alone", "has_cabin"]
ENG_CATEGORICAL = ["pclass", "sex", "embarked", "title", "deck"]

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
     ENG_NUMERIC),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", min_frequency=10)), ENG_CATEGORICAL),
])


def make_model(clf) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),   # Part 3 insights
        ("preprocess", preprocess),                        # impute, scale, one-hot (fitted on train only)
        ("model", clf),
    ])


print(make_model(LogisticRegression()))
note("""Why a Pipeline? Every learned step (median for imputation, scaling means,
category lists) is fitted ONLY on the training folds - no leakage - and the
whole thing is saved as ONE object, so production applies identical steps.
add_indicator=True adds 'age_was_missing' because missingness is informative.""")

step("5.3  Compare candidates with 5-fold cross-validation (TRAIN only)")
candidates = {
    "dummy (majority)": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=2000, C=1.0),
    "random forest": RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=3,
                                            random_state=RANDOM_STATE, n_jobs=-1),
}
cv_rows = []
for name, clf in candidates.items():
    res = cross_validate(make_model(clf), X_train, y_train, cv=cv,
                         scoring=["accuracy", "f1", "roc_auc"], return_train_score=True)
    cv_rows.append({"model": name,
                    "cv_accuracy": res["test_accuracy"].mean(),
                    "+/-": res["test_accuracy"].std(),
                    "cv_f1": res["test_f1"].mean(),
                    "cv_roc_auc": res["test_roc_auc"].mean(),
                    "train_accuracy": res["train_accuracy"].mean()})
cv_table = pd.DataFrame(cv_rows)
print(cv_table.round(3).to_string(index=False))
note("""Both ML models beat the best rule. A gap between train_accuracy and
cv_accuracy shows overfitting. When two models are within the +/- band,
prefer the simpler, more explainable one (logistic regression).""")

step("5.4  The LEAKAGE TRAP: what if we had kept 'boat' and 'body'?")
X_leaky_train = df.loc[X_train.index, FEATURE_SOURCE_COLS + LEAKY].copy()
X_leaky_train["boat"] = X_leaky_train["boat"].notna().astype(int)        # 'got into a lifeboat'
X_leaky_train["body"] = X_leaky_train["body"].notna().astype(int)        # 'body was recovered'
leaky_pre = ColumnTransformer([("leak", "passthrough", LEAKY),
                               ("rest", preprocess, ENG_NUMERIC + ENG_CATEGORICAL)])
leaky_model = Pipeline([("features", FunctionTransformer(add_features)), ("preprocess", leaky_pre),
                        ("model", LogisticRegression(max_iter=2000))])
leaky_acc = cross_val_score(leaky_model, X_leaky_train, y_train, cv=cv, scoring="accuracy").mean()
show("CV accuracy WITH boat/body", leaky_acc)
show("CV accuracy without (honest)", cv_table.loc[cv_table.model == "logistic regression", "cv_accuracy"].iloc[0])
note("""~97% looks like a breakthrough - but 'boat' simply records who survived.
At boarding time this information does not exist. Too good to be true = leakage.""")

step("5.5  FINAL evaluation: open the sealed test set ONCE")
final_model = make_model(LogisticRegression(max_iter=2000)).fit(X_train, y_train)
proba = final_model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
baseline_test = accuracy_score(y_test, rules["rung 1: women survive"].predict(X_test))
show("Test accuracy - 'women survive' rule", baseline_test)
show("Test accuracy - logistic regression", accuracy_score(y_test, pred))
show("Test ROC-AUC", roc_auc_score(y_test, proba))
cm = pd.DataFrame(confusion_matrix(y_test, pred), index=["actual died", "actual survived"],
                  columns=["pred died", "pred survived"])
print(cm.to_string())
print(classification_report(y_test, pred, target_names=["died", "survived"], digits=3))

step("5.6  What did the model learn? (coefficients = log-odds effects)")
names = final_model.named_steps["preprocess"].get_feature_names_out()
coefs = pd.Series(final_model.named_steps["model"].coef_[0], index=names).sort_values()
print(pd.concat([coefs.head(5), coefs.tail(5)]).round(3).to_string())
note("""Positive = raises survival odds, negative = lowers them. Title_Mr is the
strongest negative signal, 3rd class pushes down; title_Master (boys) and
sex_female push up.
Coefficients show ASSOCIATION, not causation (Lab 12).""")

step("5.7  Error analysis: WHO does the model get wrong?")
errors = X_test.assign(actual=y_test, p_survive=proba.round(2), pred=pred)
errors = add_features(errors)
wrong = errors[errors.actual != errors.pred]
cols_show = ["pclass", "sex", "age", "title", "family_size", "fare", "p_survive", "actual"]
print("  Survived, but model said 'died' (most confident misses):")
print(wrong[wrong.actual == 1][cols_show].sort_values("p_survive").head(5).to_string())
print("\n  Died, but model said 'survived':")
print(wrong[wrong.actual == 0][cols_show].sort_values("p_survive", ascending=False).head(5).to_string())
seg = errors.assign(correct=errors.actual == errors.pred).groupby(["sex", "pclass"]).correct.mean()
print("\n  Accuracy by segment:\n" + seg.round(3).to_string())
note("""Typical misses: 3rd-class women who died (the model trusts 'female') and
men who survived (rare, hard to predict from boarding data). The weakest
segments are 3rd-class women and 1st-class men - exactly where survival was
closest to a coin flip. Per-segment accuracy shows WHERE to invest next
(e.g. family/ticket-group features), not just a fancier algorithm.""")

step("5.8  Save the model and a metrics report (the deliverable)")
model_path = OUT_DIR / "titanic_baseline_v1.joblib"
joblib.dump(final_model, model_path)
report = {
    "model": "titanic_baseline_v1 (logistic regression pipeline)",
    "features_used": FEATURE_SOURCE_COLS,
    "excluded_leaky": LEAKY,
    "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
    "baseline_rule_test_accuracy": round(baseline_test, 4),
    "test_accuracy": round(accuracy_score(y_test, pred), 4),
    "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
    "cv_table": cv_table.round(4).to_dict(orient="records"),
}
(OUT_DIR / "titanic_metrics_v1.json").write_text(json.dumps(report, indent=2))
show("Model saved", f"outputs/{model_path.name}")
show("Report saved", "outputs/titanic_metrics_v1.json")

step("5.9  Predict for NEW passengers (as if they were boarding today)")
new_passengers = pd.DataFrame([
    {"name": "Dawson, Mr. Jack", "pclass": 3, "sex": "male", "age": 20, "sibsp": 0, "parch": 0,
     "fare": 7.9, "embarked": "S", "cabin": np.nan},
    {"name": "Bukater, Miss. Rose", "pclass": 1, "sex": "female", "age": 17, "sibsp": 0, "parch": 1,
     "fare": 211.3, "embarked": "S", "cabin": "B52"},
    {"name": "Smith, Master. Tom", "pclass": 2, "sex": "male", "age": 6, "sibsp": 1, "parch": 2,
     "fare": 30.0, "embarked": "C", "cabin": np.nan},
])
reloaded = joblib.load(model_path)
new_passengers["p_survive"] = reloaded.predict_proba(new_passengers[FEATURE_SOURCE_COLS])[:, 1].round(3)
print(new_passengers[["name", "pclass", "sex", "age", "fare", "p_survive"]].to_string(index=False))
note("""The reloaded pipeline does EVERYTHING itself: extracts the title, builds
family size, imputes, encodes, scales, predicts. That is what 'a model'
means in production - the whole recipe, not just the algorithm.""")

banner("LAB COMPLETE - what you built")
print("""  [x] Explored a real dataset with pandas (shape, types, stats, missingness, target)
  [x] Classified every column: numerical / categorical / text / ID / target / LEAKY
  [x] EDA: survival patterns, interactions, informative missingness, engineered features
  [x] Stratified, verified train/test split with the test set sealed
  [x] Baseline ladder: majority -> rules -> logistic regression -> random forest (CV)
  [x] Caught a leakage trap (boat/body) and evaluated honestly on the test set
  [x] Error analysis by segment, saved model + metrics report, predicted new passengers""")
