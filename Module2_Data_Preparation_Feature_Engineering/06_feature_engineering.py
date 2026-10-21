"""
LAB 06 - Feature Engineering: Creating Domain-Specific Features
===============================================================

DEFINITION
    Feature engineering = turning raw columns into inputs that make the pattern
    EASY for the model to see. It is where domain knowledge enters the model.
    "Applied machine learning is basically feature engineering." - Andrew Ng

THE FEATURE ENGINEER'S TOOLBOX
    ratios & differences    payment / income, loan / income, years_employed / age
    flags & thresholds      utilisation > 80%, thin credit file
    datetime parts          hour, weekday, month, "days since"; CYCLICAL sin/cos encoding
    binning & splines       non-linear effects (age < 25 is risky) for linear models
    interactions            A x B when the effect of A depends on B
    aggregations            per-customer counts / sums / trends over TIME WINDOWS
    text                    keywords, length, embeddings (Lab 11)

    Golden rule: every feature must be computable AT PREDICTION TIME, from data
    available BEFORE the moment of prediction (point-in-time correctness).

LEVELS
    EASY   : Ratio features - the domain formula a model cannot easily invent.
    MEDIUM : Time & non-linearity - cyclical hour encoding, binning/splines, interactions.
    HARD   : Behavioural features from a 600k-row transaction log, built POINT-IN-TIME
             (and what happens when they are not).
"""
# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, OneHotEncoder, SplineTransformer,
                                   StandardScaler)

from ml_lab_utils import (HAS_MPL, LOAN_NOMINAL, LOAN_NUMERIC, LOAN_TARGET, RANDOM_STATE, banner,
                          make_loan_applications, note, plt, save_fig, show, step)
from preprocessing import LoanFeatureBuilder

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): Ratio features
# ============================================================================
banner("The feature a credit analyst computes first", "EASY")
loans = make_loan_applications(n=12000, seed=61)
tr, te = train_test_split(loans, test_size=0.3, stratify=loans[LOAN_TARGET], random_state=RANDOM_STATE)
fb = LoanFeatureBuilder().fit(tr)
tr_f, te_f = fb.transform(tr), fb.transform(te)

step("1.1  The annuity formula -> monthly payment -> payment-to-income (PTI)")
print(tr_f[["annual_income", "loan_amount", "interest_rate", "loan_term_months",
            "monthly_payment", "payment_to_income"]].head(4).round(3).to_string())


def single_auc(col):
    x = te_f[col].fillna(tr_f[col].median())
    a = roc_auc_score(te_f[LOAN_TARGET], x)
    return max(a, 1 - a)


print(pd.Series({c: single_auc(c) for c in ["annual_income", "loan_amount", "interest_rate",
                                            "payment_to_income", "loan_to_income"]},
                name="AUC of the feature ALONE").round(3).to_string())
note("""Income alone and loan size alone barely beat a coin flip (0.5). Their RATIO,
adjusted for rate and term, is far stronger. (interest_rate scores well too,
but for a different reason: the lender SET it from the credit score - it is the
output of an existing risk model, a proxy. Worth a discussion: is it safe to
learn from the old model's decisions?)""")

step("1.2  Raw features vs raw + engineered, for a simple and a complex model")


def pipe(model, num):
    prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                    StandardScaler()), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), LOAN_NOMINAL)])
    return make_pipeline(prep, model)


rows = {}
for feat_name, num in [("raw columns", LOAN_NUMERIC), ("raw + 6 engineered", LOAN_NUMERIC + LoanFeatureBuilder.NEW)]:
    for m_name, m in [("Logistic", LogisticRegression(max_iter=5000)),
                      ("Boosting", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                                                  random_state=RANDOM_STATE))]:
        p = pipe(m, num).fit(tr_f, tr_f[LOAN_TARGET])
        rows[(feat_name, m_name)] = roc_auc_score(te_f[LOAN_TARGET], p.predict_proba(te_f)[:, 1])
print(pd.Series(rows).unstack().loc[["raw columns", "raw + 6 engineered"], ["Logistic", "Boosting"]]
      .round(4).to_string())
note("""Engineered features lift BOTH models. Boosting can approximate a ratio with
many axis-aligned splits, but that costs data and depth; a linear model cannot
represent a ratio at all. One line of domain knowledge is worth more than
switching algorithms - and it is EXPLAINABLE ('payment is 35% of income').""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Time, cycles and non-linearity
# ============================================================================
banner("Datetime, cycles, bins and interactions", "MEDIUM")

step("2.1  Cyclical time: card fraud peaks around 2-3 AM")
n = 20000
hour = rng.integers(0, 24, n)
dist_from_3am = np.minimum(np.abs(hour - 2.5), 24 - np.abs(hour - 2.5))
fraud = rng.binomial(1, 1 / (1 + np.exp(-(-4 + 2.2 * np.exp(-dist_from_3am / 2.5)))))
H = pd.DataFrame({"hour": hour})
Htr, Hte, ftr, fte = train_test_split(H, fraud, test_size=0.3, random_state=RANDOM_STATE, stratify=fraud)


def sin_cos(X):
    a = 2 * np.pi * np.asarray(X, float) / 24
    return np.column_stack([np.sin(a), np.cos(a)])


encodings = {
    "hour as a number (0-23)": make_pipeline(StandardScaler(), LogisticRegression()),
    "sin/cos of hour": make_pipeline(FunctionTransformer(sin_cos), LogisticRegression()),
    "one-hot hour (24 columns)": make_pipeline(OneHotEncoder(), LogisticRegression()),
    "periodic spline": make_pipeline(SplineTransformer(n_knots=7, degree=3, extrapolation="periodic",
                                                       knots=np.linspace(0, 24, 7).reshape(-1, 1)),
                                     LogisticRegression()),
}
print(pd.Series({k: roc_auc_score(fte, v.fit(Htr, ftr).predict_proba(Hte)[:, 1])
                 for k, v in encodings.items()}, name="fraud AUC").round(4).to_string())
note("""As a number, 23:00 and 01:00 are 22 units apart, yet they are 2 hours apart
in reality - a linear model cannot express 'risky around 2-3 AM'. sin/cos puts
hours on a circle so midnight wraps around. One-hot works too (24 columns);
periodic splines are a smooth middle ground. Same idea for weekday, month and
compass direction.""")

step("2.2  Non-linear effects: credit utilisation has a cliff at 80%")
print(tr_f.groupby(pd.cut(tr_f.credit_utilization, [0, .2, .4, .6, .7, .8, .9, 1.2]), observed=True)
      [LOAN_TARGET].mean().rename("default rate by utilisation band").round(3).to_string())
util_encodings = {
    "utilisation as a number": FunctionTransformer(),
    "utilisation in 8 quantile bins": KBinsDiscretizer(n_bins=8, encode="onehot-dense", strategy="quantile",
                                                       quantile_method="averaged_inverted_cdf"),
    "utilisation as a spline": SplineTransformer(n_knots=6, degree=3),
}
base_num = ["payment_to_income", "credit_score", "delinquencies_2y", "age"]
res = {}
for name, enc in util_encodings.items():
    prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), base_num),
                              ("util", make_pipeline(enc, StandardScaler()), ["credit_utilization"])])
    p = make_pipeline(prep, LogisticRegression(max_iter=5000)).fit(tr_f, tr_f[LOAN_TARGET])
    res[name] = roc_auc_score(te_f[LOAN_TARGET], p.predict_proba(te_f)[:, 1])
print(pd.Series(res, name="test AUC").round(4).to_string())
note("""Risk more than doubles past 80% utilisation (a known credit-bureau
pattern). A straight line cannot jump; bins and splines let a LINEAR model
bend. Binning is easy to explain to a credit committee; splines are smoother;
a domain FLAG (utilisation > 0.8, in LoanFeatureBuilder) is simplest of all.
Bin edges are learned in fit -> on TRAIN only.""")

step("2.3  Interactions: high utilisation is worse when you already have delinquencies")
tr_i, te_i = tr_f.copy(), te_f.copy()
for d in (tr_i, te_i):
    d["util_x_delinq"] = d.credit_utilization * d.delinquencies_2y
    d["pti_x_thin_file"] = d.payment_to_income * d.thin_file
for name, extra in [("no interactions", []), ("+ 2 domain interactions", ["util_x_delinq", "pti_x_thin_file"])]:
    p = pipe(LogisticRegression(max_iter=5000), LOAN_NUMERIC + LoanFeatureBuilder.NEW + extra)
    p.fit(tr_i, tr_i[LOAN_TARGET])
    show(f"Logistic, {name}", roc_auc_score(te_i[LOAN_TARGET], p.predict_proba(te_i)[:, 1]))
note("""Honest result: tiny or no gain - these interactions are not part of the
true mechanism in this dataset. Engineered features are HYPOTHESES. Keep the
ones the data supports (validate on held-out data), drop the rest. Adding
PolynomialFeatures blindly multiplies columns and overfitting risk (Lab 08).""")

# %% =========================================================================
# LEVEL 3 (HARD): Behavioural features from an event log - point-in-time
# ============================================================================
banner("Aggregations over time windows, built point-in-time", "HARD")
note("""Business question: on day 270, which credit-card customers will default in
the next 90 days? We have a raw TRANSACTION LOG for days 0-364. Defaulted
accounts are frozen, so they have no transactions AFTER their default date.""")

N_CUST, DAYS, SNAPSHOT = 3000, 365, 270
z = rng.normal(0, 1, N_CUST)                                        # hidden risk
p_def = 1 / (1 + np.exp(-(-2.2 + 1.1 * z)))
defaults = rng.random(N_CUST) < p_def
default_day = np.where(defaults, rng.integers(SNAPSHOT + 1, SNAPSHOT + 91, N_CUST), 10_000)
credit_score = np.clip(690 - 35 * z + rng.normal(0, 45, N_CUST), 300, 850).round()
customers = pd.DataFrame({"customer_id": np.arange(N_CUST), "credit_score": credit_score,
                          "credit_limit": rng.choice([2000, 5000, 10000], N_CUST),
                          "defaulted_next_90d": defaults.astype(int)})

day = np.arange(DAYS)
pre_default = (day[None, :] > default_day[:, None] - 75) & (day[None, :] <= default_day[:, None])
rate = 0.6 * rng.lognormal(0, 0.3, N_CUST)[:, None] * np.where(pre_default, 1.6, 1.0)   # habits vary
rate = np.where(day[None, :] > default_day[:, None], 0, rate)         # frozen after default
counts = rng.poisson(rate)
cust_idx, day_idx = np.nonzero(counts)
reps = counts[cust_idx, day_idx]
cust_idx, day_idx = np.repeat(cust_idx, reps), np.repeat(day_idx, reps)
p_cash = 0.03 * np.exp(0.5 * z[cust_idx]) * np.where(pre_default[cust_idx, day_idx], 3.0, 1.0)
is_cash = rng.random(len(cust_idx)) < np.clip(p_cash, 0, 0.9)
amount = np.where(is_cash, rng.lognormal(5.3, 0.5, len(cust_idx)), rng.lognormal(3.6, 0.8, len(cust_idx)))
events = pd.DataFrame({"customer_id": cust_idx, "day": day_idx, "amount": amount.round(2),
                       "type": np.where(is_cash, "cash_advance", "purchase")})
show("Transaction log rows", f"{len(events):,}")
print(events.head(5).to_string(index=False))


def behaviour_features(ev: pd.DataFrame, as_of: int) -> pd.DataFrame:
    """Windowed aggregates using ONLY events strictly before `as_of`."""
    past = ev[ev.day < as_of]
    feats = pd.DataFrame(index=pd.Index(range(N_CUST), name="customer_id"))
    for w in (30, 90):
        win = past[past.day >= as_of - w]
        g = win.groupby("customer_id")
        feats[f"txn_count_{w}d"] = g.size()
        feats[f"spend_{w}d"] = g.amount.sum()
        feats[f"cash_adv_{w}d"] = win[win.type == "cash_advance"].groupby("customer_id").size()
    feats = feats.fillna(0)
    feats["spend_trend_30_vs_90"] = feats.spend_30d / (feats.spend_90d / 3 + 1)
    feats["days_since_last_txn"] = as_of - past.groupby("customer_id").day.max()
    return feats.fillna(DAYS).reset_index()


pit = customers.merge(behaviour_features(events, as_of=SNAPSHOT), on="customer_id")
naive = customers.merge(behaviour_features(events, as_of=DAYS), on="customer_id")   # "latest data" = future
BEH = [c for c in pit.columns if c not in customers.columns]

idx_tr, idx_te = train_test_split(np.arange(N_CUST), test_size=0.3, random_state=RANDOM_STATE,
                                  stratify=customers.defaulted_next_90d)


def fit_auc(df_train, df_eval, cols):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    m.fit(df_train.loc[idx_tr, cols], df_train.loc[idx_tr, "defaulted_next_90d"])
    return roc_auc_score(df_eval.loc[idx_te, "defaulted_next_90d"],
                         m.predict_proba(df_eval.loc[idx_te, cols])[:, 1]), m


auc_static, _ = fit_auc(pit, pit, ["credit_score", "credit_limit"])
auc_pit, m_pit = fit_auc(pit, pit, ["credit_score", "credit_limit"] + BEH)
auc_naive_offline, m_naive = fit_auc(naive, naive, ["credit_score", "credit_limit"] + BEH)
auc_naive_prod = roc_auc_score(pit.loc[idx_te, "defaulted_next_90d"],
                               m_naive.predict_proba(pit.loc[idx_te, ["credit_score", "credit_limit"] + BEH])[:, 1])
print(pd.DataFrame({
    "feature set": ["static only (score, limit)", "+ behaviour, POINT-IN-TIME (as of day 270)",
                    "+ behaviour, NAIVE (computed on all data) - offline",
                    "+ NAIVE model, scored on day 270 (= production)"],
    "AUC": [auc_static, auc_pit, auc_naive_offline, auc_naive_prod]}).round(4).to_string(index=False))
note("""The naive features were computed 'as of the latest data' - day 365 - so a
defaulted customer shows 'no transactions in 30 days' (account frozen AFTER
default). The model learns 'silence = default': near-perfect offline, much
worse on day 270 when nobody is frozen yet. Point-in-time features give an
honest lift over static data: rising spend and cash advances BEFORE default.
In production this is what a FEATURE STORE's point-in-time join guarantees.""")

step("3.1  Which behavioural features carry the signal? (point-in-time model)")
imp = permutation_importance(m_pit, pit.loc[idx_te, ["credit_score", "credit_limit"] + BEH],
                             pit.loc[idx_te, "defaulted_next_90d"], scoring="roc_auc",
                             n_repeats=10, random_state=RANDOM_STATE)
print(pd.Series(imp.importances_mean, index=["credit_score", "credit_limit"] + BEH)
      .sort_values(ascending=False).round(4).to_string())
note("""Surprise: once behaviour is known, the bureau credit score adds ~nothing -
alone it gave AUC 0.67, but cash-advance counts already carry the same hidden
risk, more freshly. Importance is CONDITIONAL on the other features. That is
the topic of the next lab: redundant features and feature selection.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    d_ids = customers.customer_id[customers.defaulted_next_90d == 1].to_numpy()
    daily = events.assign(grp=np.where(events.customer_id.isin(d_ids), "will default", "will not default"))
    curve = daily.groupby(["grp", "day"]).size().unstack(0)
    curve = curve / curve.iloc[:60].mean()
    curve.rolling(7).mean().plot(ax=ax)
    ax.legend(title=None)
    ax.axvline(SNAPSHOT, color="black", ls="--")
    ax.text(SNAPSHOT + 2, ax.get_ylim()[1] * 0.9, "prediction date (day 270)", fontsize=8)
    ax.set_ylabel("transactions (relative)")
    ax.set_title("Behaviour changes BEFORE default; activity stops AFTER it (the leak)")
    save_fig(fig, "06_hard_point_in_time")

banner("KEY TAKEAWAYS")
print("""  1. The best features encode domain formulas: ratios, flags, payment-to-income.
  2. Encode time as cycles (sin/cos, splines); let linear models bend with bins/splines.
  3. Engineered features are hypotheses - keep only what held-out data supports.
  4. Behavioural aggregates over time windows are often the strongest signals.
  5. Compute every feature AS OF the prediction moment - never 'as of the latest data'.""")
