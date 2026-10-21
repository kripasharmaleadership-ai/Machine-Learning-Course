"""
LAB 01 - Why Data Quality Matters More Than Algorithm Complexity
================================================================

THE CLAIM
    "Better data beats fancier algorithms."  (Peter Norvig, Andrew Ng's "data-centric AI")
    In industry, most of the gains after the first model come from fixing data,
    not from switching models. This lab tests that claim with numbers.

THE SIX DIMENSIONS OF DATA QUALITY
    Completeness  - are values present?              (missing income, blank scores)
    Validity      - do values obey the rules?        (age 999, utilisation 4,500%)
    Consistency   - same thing, same representation? ("Salaried" vs " SALARIED")
    Uniqueness    - is each entity recorded once?    (ETL loaded a batch twice)
    Accuracy      - does the value match reality?    (income typed in cents; WRONG LABELS)
    Timeliness    - is it fresh enough to use?       (last month's bureau scores)

LEVELS
    EASY   : Data-quality scorecard of a raw export, then "dirty data + boosting" vs
             "clean data + logistic regression".
    MEDIUM : Label noise x model complexity grid. Random vs SYSTEMATIC label errors.
    HARD   : Fixed engineering budget: spend it on hyperparameter tuning (model-centric)
             or on finding & fixing bad labels (data-centric)? We measure both.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml_lab_utils import (EDUCATION_LEVELS, HAS_MPL, LOAN_NOMINAL, LOAN_NUMERIC, LOAN_TARGET,
                          RANDOM_STATE, banner, make_loan_applications, note, plt, save_fig, show,
                          step)
from preprocessing import (ColumnRule, DataContract, LoanFeatureBuilder, normalize_category,
                           parse_dates, parse_money)

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): Quality scorecard, then dirty+complex vs clean+simple
# ============================================================================
banner("What does a raw production export really look like?", "EASY")
raw = make_loan_applications(messy=True)              # training extract (2024-2025)
future = make_loan_applications(seed=2026, messy=True).drop_duplicates()   # next quarter's applicants
truth = make_loan_applications(with_truth=True)       # the same rows, before corruption
print(raw.sample(6, random_state=3).T.to_string())

step("1.1  Data-quality scorecard (the six dimensions, measured)")
contract = DataContract(
    rules={
        "age": ColumnRule("number", min=18, max=100, max_missing=0.02),
        "annual_income": ColumnRule("number", min=5_000, max=2_000_000, max_missing=0.05),
        "credit_score": ColumnRule("number", min=300, max=850, max_missing=0.05),
        "credit_utilization": ColumnRule("number", min=0, max=1.5),
        "loan_amount": ColumnRule("number", min=500, max=100_000),
        "employment_type": ColumnRule("category", allowed=["Salaried", "Self-employed", "Contract",
                                                           "Unemployed", "Retired"]),
        "home_ownership": ColumnRule("category", allowed=["Rent", "Mortgage", "Own"]),
    },
    unique_key="application_id",
)
issues = contract.validate(raw)
print(issues.to_string(index=False))
note("""Every line is a silent bug waiting to happen. 'annual_income' is not even
numeric - half the values are '$54,300' strings. An algorithm cannot fix this:
pd.to_numeric() will quietly turn every one of them into NaN.""")

step("1.2  Two teams, two strategies")


def team_a_quick(df):
    """Team A: 'just get it into the model' - coerce, one-hot as-is, ship it."""
    out = df.copy()
    for c in LOAN_NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")     # '$54,300' -> NaN, silently
    return out


EMP_MAP = {"self employed": "Self-employed", "salaried": "Salaried", "contract": "Contract",
           "contractor": "Contract", "unemployed": "Unemployed", "not employed": "Unemployed",
           "retired": "Retired"}


def team_b_clean(df):
    """Team B: ~15 lines of boring, careful cleaning (each line = one quality rule)."""
    out = df.copy()
    out["annual_income"] = parse_money(out.annual_income)                         # validity
    out["age"] = out.age.where(out.age.between(18, 100))                           # sentinels 0/999
    out["credit_score"] = out.credit_score.where(out.credit_score.between(300, 850))  # 0 = "no score"
    out["credit_utilization"] = np.where(out.credit_utilization > 1.5,            # entered as %
                                         out.credit_utilization / 100, out.credit_utilization)
    out["employment_type"] = normalize_category(out.employment_type, EMP_MAP)      # consistency
    for c in ["home_ownership", "state"]:
        out[c] = normalize_category(out[c])
    out["application_date"] = parse_dates(out.application_date)
    key = ["age", "annual_income", "loan_amount", "loan_purpose", "application_date", "state"]
    out = out.drop_duplicates().drop_duplicates(subset=key)                        # uniqueness
    return out


def make_model(kind, num_cols):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), LOAN_NOMINAL),
        ("edu", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OrdinalEncoder(categories=[EDUCATION_LEVELS])), ["education"]),
    ], sparse_threshold=0)
    clf = {"Logistic regression": LogisticRegression(max_iter=2000, C=0.5),
           "Gradient boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE)}[kind]
    return Pipeline([("prep", prep), ("clf", clf)])


results = {}
for team, fn in [("A: raw data, coerced", team_a_quick), ("B: cleaned data", team_b_clean)]:
    tr, te = fn(raw), fn(future)
    for kind in ["Logistic regression", "Gradient boosting"]:
        m = make_model(kind, LOAN_NUMERIC).fit(tr, tr[LOAN_TARGET])
        results[(team, kind)] = roc_auc_score(te[LOAN_TARGET], m.predict_proba(te)[:, 1])
    show(f"Team {team[0]} training rows", len(tr))

step("1.3  ...and Team B adds domain knowledge (payment-to-income etc., Lab 06)")
fb = LoanFeatureBuilder()
tr_b, te_b = team_b_clean(raw), team_b_clean(future)
tr_b = fb.fit_transform(tr_b)
te_b = fb.transform(te_b)
m = make_model("Logistic regression", LOAN_NUMERIC + LoanFeatureBuilder.NEW).fit(tr_b, tr_b[LOAN_TARGET])
results[("B: cleaned + domain features", "Logistic regression")] = roc_auc_score(
    te_b[LOAN_TARGET], m.predict_proba(te_b)[:, 1])
table = pd.Series(results).unstack().round(3)
table = table.reindex(["A: raw data, coerced", "B: cleaned data", "B: cleaned + domain features"])
print(table[["Logistic regression", "Gradient boosting"]].to_string(na_rep="-"))
show("Oracle AUC (true probabilities - the ceiling)",
     roc_auc_score(truth[LOAN_TARGET], truth["_true_default_prob"]))
note("""Read the table diagonally: a SIMPLE model on CLEAN data with ONE good
domain feature beats a powerful boosting model on raw data. Boosting is robust
to many things, but it cannot recover information that parsing destroyed
(income became NaN for half the rows) and cannot invent ratios the data
never contained in a usable form.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Label noise x model complexity
# ============================================================================
banner("Wrong labels: the quality problem no algorithm sees", "MEDIUM")
clean = make_loan_applications(n=8000, seed=11)
test = make_loan_applications(n=8000, seed=12)
fb = LoanFeatureBuilder().fit(clean)
Xtr, Xte = fb.transform(clean), fb.transform(test)
NUM = LOAN_NUMERIC + LoanFeatureBuilder.NEW
y_clean, y_test = clean[LOAN_TARGET].to_numpy(), test[LOAN_TARGET].to_numpy()

MODELS = {
    "Logistic": lambda: LogisticRegression(max_iter=2000, C=0.5),
    "Random forest": lambda: RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1,
                                                    random_state=RANDOM_STATE),
    "Boosting": lambda: HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "Neural net": lambda: MLPClassifier((64, 32), alpha=1e-3, max_iter=300, early_stopping=True,
                                        random_state=RANDOM_STATE),
}


def fit_eval(model_name, y_train):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOAN_NOMINAL),
    ])
    pipe = make_pipeline(prep, MODELS[model_name]()).fit(Xtr, y_train)
    return roc_auc_score(y_test, pipe.predict_proba(Xte)[:, 1])


step("2.1  Randomly flip a share of the training labels; test on CORRECT labels")
noise_levels = [0.0, 0.1, 0.2, 0.3]
grid = pd.DataFrame(index=list(MODELS), columns=[f"{p:.0%} noise" for p in noise_levels], dtype=float)
for p in noise_levels:
    flip = rng.random(len(y_clean)) < p
    y_noisy = np.where(flip, 1 - y_clean, y_clean)
    for name in MODELS:
        grid.loc[name, f"{p:.0%} noise"] = fit_eval(name, y_noisy)
print(grid.round(3).to_string())
note("""Columns hurt more than rows: moving one column right (more label noise)
costs more than any move between models. The spread between the best and the
worst MODEL is small; the spread between clean and 30%-noisy DATA is large.""")

step("2.2  Systematic noise: a collections system missed defaults in 3 states")
sys_noise = y_clean.copy()
broken = clean.state.isin(["TX", "FL", "GA"]).to_numpy() & (y_clean == 1)
sys_noise[broken] = 0                           # defaults never recorded for these states
show("Labels changed", f"{broken.sum()} ({broken.mean():.1%} of rows)")
te_mask = test.state.isin(["TX", "FL", "GA"]).to_numpy()
y_test_broken = np.where(te_mask & (y_test == 1), 0, y_test)   # validation labels: same broken system
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOAN_NOMINAL)])
pipe = make_pipeline(prep, LogisticRegression(max_iter=2000, C=0.5)).fit(Xtr, sys_noise)
p_te = pipe.predict_proba(Xte)[:, 1]
show("AUC the team SEES (labels from same system)", roc_auc_score(y_test_broken, p_te))
show("AUC in REALITY (true outcomes)", roc_auc_score(y_test, p_te))
show("Predicted default rate, TX/FL/GA applicants", f"{p_te[te_mask].mean():.1%}")
show("ACTUAL default rate, TX/FL/GA applicants", f"{y_test[te_mask].mean():.1%}")
note("""Systematic errors are far more dangerous than random ones, because your
validation labels come from the SAME broken system. The team sees an excellent
AUC; reality is much worse. The model has LEARNED that three states are almost
risk-free and will approve risky loans there. A more powerful algorithm would
learn this bias even more faithfully. Only a data check finds it, for example
'default rate by state over time'.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iloc[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Test AUC: model choice (rows) vs label quality (columns)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_fig(fig, "01_medium_noise_vs_model")

# %% =========================================================================
# LEVEL 3 (HARD): Model-centric vs data-centric - same budget
# ============================================================================
banner("You have 2 weeks. Tune the model, or fix the labels?", "HARD")
p_noise = 0.20
flip = rng.random(len(y_clean)) < p_noise
y_noisy = np.where(flip, 1 - y_clean, y_clean)
show("Training labels that are wrong", f"{flip.sum()} of {len(flip)}")


# The team's existing, sensible production config (not sklearn's defaults):
TEAM_CONFIG = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, max_iter=200)


def boosting_pipe(**params):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOAN_NOMINAL)])
    return Pipeline([("prep", prep), ("clf", HistGradientBoostingClassifier(
        random_state=RANDOM_STATE, **{**TEAM_CONFIG, **params}))])


def test_auc(model):
    return roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])


auc_base = test_auc(boosting_pipe().fit(Xtr, y_noisy))
show("Team config, noisy labels", auc_base)

step("3.1  MODEL-CENTRIC: randomized hyperparameter search (30 configs x 5 folds)")
SPACE = {"clf__learning_rate": [0.02, 0.05, 0.1, 0.2], "clf__max_leaf_nodes": [7, 15, 31, 63],
         "clf__min_samples_leaf": [20, 50, 100, 200], "clf__l2_regularization": [0, 0.1, 1, 10],
         "clf__max_iter": [100, 200, 400]}


def tune(y):
    return RandomizedSearchCV(boosting_pipe(), SPACE, n_iter=30, scoring="roc_auc",
                              cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                              random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, y)


search_noisy = tune(y_noisy)
auc_tuned = test_auc(search_noisy)
show("Tuned config, noisy labels", auc_tuned)

step("3.2  DATA-CENTRIC: find suspicious labels with out-of-fold predictions")
note("""Idea behind 'confident learning' (Northcutt et al., cleanlab): train with
cross-validation, and for each row ask 'how likely is the label we HAVE, according
to a model that never saw this row?'. Very unlikely labels are review candidates.
Budget: reviewers can check 1,200 rows (15%) in two weeks.""")
oof = cross_val_predict(make_pipeline(
    ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
                       ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOAN_NOMINAL)]),
    LogisticRegression(max_iter=2000, C=0.5)), Xtr, y_noisy,
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), method="predict_proba")
p_given = np.where(y_noisy == 1, oof[:, 1], oof[:, 0])     # model's belief in the label we have
budget = 1200
review = np.argsort(p_given)[:budget]                      # least believable labels first
precision = flip[review].mean()
y_fixed = y_noisy.copy()
y_fixed[review] = y_clean[review]                          # humans check & correct these rows
show("Share of reviewed rows that were truly wrong", f"{precision:.1%}  (random review: {p_noise:.0%})")
show("Wrong labels left after review", f"{(y_fixed != y_clean).sum()} (was {flip.sum()})")
auc_fixed = test_auc(boosting_pipe().fit(Xtr, y_fixed))
show("Team config, reviewed labels", auc_fixed)

step("3.3  The full picture: tuning x label quality")
auc_both = test_auc(tune(y_fixed))
auc_perfect = test_auc(boosting_pipe().fit(Xtr, y_clean))
summary = pd.DataFrame({
    "strategy": ["Baseline: team config, noisy labels", "Model-centric: 30-config tuning",
                 "Data-centric: review 15% of labels", "Both: reviewed labels + tuning",
                 "Upper bound: all labels correct"],
    "test_auc": [auc_base, auc_tuned, auc_fixed, auc_both, auc_perfect]})
summary["gain_vs_baseline"] = summary.test_auc - auc_base
print(summary.round(4).to_string(index=False))
print(f"  Best tuned params on NOISY labels: { {k[5:]: v for k, v in search_noisy.best_params_.items()} }")
note("""Read it like a principal engineer would:
 - Reviewing just 15% of labels gets close to the 'all labels correct' bound, and the
   review queue was ~4x more efficient than random checking.
 - Tuning ALSO helped, but look at what it chose: heavy regularisation (big leaves,
   few splits). It is a model-side workaround for a data problem - it learns to
   ignore detail because the detail is noisy.
 - The two stack. But the fixed labels are an ASSET: they improve every future model,
   every retrain, every evaluation. The tuned config expires with this snapshot.
This is the same discipline used to build LLM fine-tuning sets: label audits,
disagreement queues, annotation guidelines.""", keep_lines=True)

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(summary.strategy[::-1], summary.test_auc[::-1],
            color=["#bab0ac", "#b07aa1", "#59a14f", "#4c78a8", "#e15759"])
    ax.set_xlim(summary.test_auc.min() - 0.02, summary.test_auc.max() + 0.01)
    for i, v in enumerate(summary.test_auc[::-1]):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center")
    ax.set_xlabel("Test AUC (on correct labels)")
    ax.set_title("Same effort: tuning the model vs fixing the data")
    save_fig(fig, "01_hard_model_vs_data_centric")

banner("KEY TAKEAWAYS")
print("""  1. Measure quality on six dimensions before modelling - a contract turns them into checks.
  2. Parsing and consistency bugs destroy information silently; no algorithm can recover it.
  3. Label noise moves AUC more than switching models does. Systematic noise teaches bias.
  4. Out-of-fold predictions turn 'check every label' into a short, high-yield review queue.
  5. Data fixes compound across every future model; tuning gains do not.""")
