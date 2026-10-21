"""
LAB 12 - Correlation vs Causation
=================================

    CORRELATION : X and Y move together.
    CAUSATION   : CHANGING X CHANGES Y.

    ML models learn CORRELATIONS - that is enough to PREDICT ("who will churn?").
    Business leaders usually want to INTERVENE ("what should we change to reduce churn?").
    That requires CAUSATION - and a model's feature importance does NOT give you that.

WHY THEY DIFFER
    1. CONFOUNDER        : a hidden third factor drives both  (heat -> ice cream AND drownings)
    2. REVERSE CAUSATION : Y causes X, not X causes Y          (loyal users install the app)
    3. SELECTION BIAS    : who got "treated" was not random    (we call the riskiest customers)
    4. COINCIDENCE       : two trends that both just go up     (spurious correlation)

LEVELS
    EASY   : Ice cream & drownings; two unrelated trends that correlate at 0.9.
    MEDIUM : "Retention calls INCREASE churn!" - selection bias and Simpson's paradox.
    HARD   : Estimate a discount's TRUE effect: naive vs regression adjustment vs
             propensity weighting vs a randomized A/B test (with confidence intervals);
             then prove a highly predictive feature can be a useless lever.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from ml_lab_utils import HAS_MPL, RANDOM_STATE, banner, note, plt, save_fig, show, step

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): confounders and coincidences
# ============================================================================
banner("Ice cream does not cause drowning", "EASY")
days = 365
temperature = 18 + 10 * np.sin(np.linspace(0, 2 * np.pi, days)) + rng.normal(0, 2, days)
ice_cream_sales = 200 + 25 * temperature + rng.normal(0, 60, days)        # caused by heat
drownings = np.clip(0.15 * temperature + rng.normal(0, 1, days), 0, None)  # caused by heat (swimming)
beach = pd.DataFrame({"temperature": temperature, "ice_cream": ice_cream_sales, "drownings": drownings})
show("corr(ice cream, drownings)", beach.ice_cream.corr(beach.drownings))


def residual(y: pd.Series, x: pd.Series) -> np.ndarray:
    """Remove the part of y explained by x (what's left after controlling for x)."""
    return y - LinearRegression().fit(x.to_frame(), y).predict(x.to_frame())


partial = np.corrcoef(residual(beach.ice_cream, beach.temperature),
                      residual(beach.drownings, beach.temperature))[0, 1]
show("Same correlation, controlling for temperature", partial)
note("""Temperature is a CONFOUNDER: it drives both. Once we hold temperature fixed
the link disappears. Banning ice cream would not save a single swimmer.
A model predicting drownings from ice cream sales would still PREDICT well -
prediction does not need causation, intervention does.""")

step("Spurious correlation: two completely independent random trends")
corrs, diff_corrs = [], []
for seed in range(200):
    r = np.random.default_rng(seed)
    a = np.cumsum(r.normal(0.1, 1, 120))   # e.g. 'monthly app downloads'
    b = np.cumsum(r.normal(0.1, 1, 120))   # e.g. 'regional cheese prices'
    corrs.append(np.corrcoef(a, b)[0, 1])
    diff_corrs.append(np.corrcoef(np.diff(a), np.diff(b))[0, 1])
corrs = np.array(corrs)
show("Pairs of UNRELATED series tested", len(corrs))
show("Share with |correlation| > 0.7 (levels)", f"{np.mean(np.abs(corrs) > 0.7):.0%}")
show("Same, on month-to-month CHANGES", f"{np.mean(np.abs(np.array(diff_corrs)) > 0.7):.0%}")
note("""Anything that trends over time correlates with anything else that trends.
Dashboards full of 'metric A tracks metric B' are often this. Correlate
CHANGES (differences), not levels, and demand a mechanism.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sc = axes[0].scatter(beach.ice_cream, beach.drownings, c=beach.temperature, cmap="coolwarm", s=10)
    axes[0].set(xlabel="ice cream sales", ylabel="drownings", title="Correlated - colour = temperature")
    fig.colorbar(sc, ax=axes[0], label="temperature")
    axes[1].hist(corrs, bins=30)
    axes[1].set(xlabel="correlation", title="200 pairs of unrelated random trends")
    save_fig(fig, "12_easy_confounder_and_spurious")

# %% =========================================================================
# LEVEL 2 (MEDIUM): selection bias and Simpson's paradox
# ============================================================================
banner("'Our retention calls INCREASE churn!' - or do they?", "MEDIUM")
n = 20_000
risk = rng.choice(["low", "medium", "high"], n, p=[0.5, 0.3, 0.2])
base_churn = pd.Series(risk).map({"low": 0.05, "medium": 0.20, "high": 0.50}).to_numpy()
# The team calls mostly HIGH-risk customers (that's their job!)
p_called = pd.Series(risk).map({"low": 0.05, "medium": 0.30, "high": 0.80}).to_numpy()
called = rng.binomial(1, p_called)
# TRUE effect: a call REDUCES churn probability by 30% (relative) for everyone
churn = rng.binomial(1, base_churn * np.where(called == 1, 0.7, 1.0))
calls = pd.DataFrame({"risk": risk, "called": called, "churned": churn})

step("Naive comparison (what the dashboard shows)")
naive = calls.groupby("called").churned.mean()
print(f"  churn rate - NOT called: {naive[0]:.1%}   |   called: {naive[1]:.1%}")
note("Called customers churn MORE. Conclusion: stop calling? That would be a disaster.")

step("Same data, split by risk segment")
seg = calls.pivot_table(index="risk", columns="called", values="churned", aggfunc="mean")
seg.columns = ["not called", "called"]
seg["call effect (pts)"] = (seg["called"] - seg["not called"]) * 100
print(seg.loc[["low", "medium", "high"]].round(3).to_string())
note("""Inside EVERY segment, calls reduce churn. Overall, the effect flips sign
because calls were aimed at high-risk customers (selection bias). This
reversal is SIMPSON'S PARADOX. Whenever 'treatment' was targeted, a naive
comparison measures WHO you targeted, not WHAT the treatment did.""")

step("Reverse causation, in one line")
note("""'Customers who use our mobile app churn 40% less - push everyone to install
it!' But maybe LOYAL customers are the ones who bother installing the app.
Loyalty -> app, not app -> loyalty. Only an experiment can separate the two.""")

# %% =========================================================================
# LEVEL 3 (HARD): estimating a causal effect - methods vs the truth
# ============================================================================
banner("What is the TRUE effect of a 15% discount on churn?", "HARD")
n = 30_000
tenure = rng.integers(1, 72, n)
monthly = rng.uniform(20, 120, n)
complaints = rng.poisson(1, n)
X = pd.DataFrame({"tenure": tenure, "monthly": monthly, "complaints": complaints})


def churn_prob(discount: np.ndarray) -> np.ndarray:
    logit = -1.0 - 0.04 * tenure + 0.02 * (monthly - 70) + 0.4 * complaints - 0.6 * discount
    return 1 / (1 + np.exp(-logit))


# Ground truth (only knowable in a simulation): average effect if EVERYONE got the discount
TRUE_ATE = float(np.mean(churn_prob(np.ones(n)) - churn_prob(np.zeros(n))))
show("TRUE average effect (simulation only)", f"{TRUE_ATE * 100:+.2f} pts")

# Observational world: sales reps hand out discounts to unhappy, expensive, new customers
prop_logit = -1.5 + 0.03 * (monthly - 70) + 0.6 * complaints - 0.03 * tenure
p_discount = 1 / (1 + np.exp(-prop_logit))
got_discount = rng.binomial(1, p_discount)
churned_obs = rng.binomial(1, churn_prob(got_discount))

estimates = {}
# 1) Naive difference
estimates["1. naive (discount vs no discount)"] = churned_obs[got_discount == 1].mean() - churned_obs[got_discount == 0].mean()
# 2) Regression adjustment: model churn ~ treatment + confounders, compare predictions
outcome_model = LogisticRegression(max_iter=2000).fit(X.assign(t=got_discount), churned_obs)
estimates["2. regression adjustment"] = np.mean(outcome_model.predict_proba(X.assign(t=1))[:, 1]
                                                - outcome_model.predict_proba(X.assign(t=0))[:, 1])
# 2b) Same method, but we FORGOT to collect complaints (one missing confounder)
partial_model = LogisticRegression(max_iter=2000).fit(X[["tenure", "monthly"]].assign(t=got_discount), churned_obs)
estimates["2b. regression adj. MISSING complaints"] = np.mean(
    partial_model.predict_proba(X[["tenure", "monthly"]].assign(t=1))[:, 1]
    - partial_model.predict_proba(X[["tenure", "monthly"]].assign(t=0))[:, 1])
# 3) Inverse propensity weighting: re-weight so treated & untreated look alike
ps = LogisticRegression(max_iter=2000).fit(X, got_discount).predict_proba(X)[:, 1].clip(0.02, 0.98)
estimates["3. inverse propensity weighting"] = (np.mean(got_discount * churned_obs / ps)
                                                - np.mean((1 - got_discount) * churned_obs / (1 - ps)))
# 4) Randomized A/B test: flip a coin for who gets the discount
ab = rng.binomial(1, 0.5, n)
churned_ab = rng.binomial(1, churn_prob(ab))
estimates["4. randomized A/B test"] = churned_ab[ab == 1].mean() - churned_ab[ab == 0].mean()

boot = []
for _ in range(500):
    i = rng.integers(0, n, n)
    t, yb = ab[i], churned_ab[i]
    boot.append(yb[t == 1].mean() - yb[t == 0].mean())
lo, hi = np.percentile(boot, [2.5, 97.5])

table = pd.DataFrame({"estimate (pts)": {k: v * 100 for k, v in estimates.items()}})
table["error vs truth (pts)"] = table["estimate (pts)"] - TRUE_ATE * 100
print(table.round(2).to_string())
show("A/B test 95% bootstrap CI", f"[{lo * 100:+.2f}, {hi * 100:+.2f}] pts")
note("""Naive: the discount appears to INCREASE churn - because reps gave it to
unhappy customers. Adjustment methods recover the truth ONLY because we
measured every confounder (tenure, price, complaints). Drop ONE confounder
(2b) and the estimate is off by almost half - with no error message. The randomized test needs no assumptions: the coin
flip breaks every hidden link. That is why A/B tests are the gold standard
- use observational methods only when experiments are impossible.""")

step("A highly PREDICTIVE feature can be a USELESS lever")
note("""Hidden network quality causes BOTH support tickets and churn. Tickets are
a SYMPTOM. The model will say 'tickets are the top churn driver'. An exec
proposes: 'make it harder to open tickets'. What happens?""")
quality = rng.normal(0, 1, n)                                        # hidden
tickets = rng.poisson(np.exp(0.3 - 0.8 * quality))                   # symptom of bad quality
p_true = 1 / (1 + np.exp(-(-1.2 - 1.0 * quality - 0.03 * (tenure - 36))))  # tickets NOT in the formula
churn_q = rng.binomial(1, p_true)
feat = pd.DataFrame({"tickets": tickets, "tenure": tenure})
model = HistGradientBoostingClassifier(max_depth=3, random_state=RANDOM_STATE).fit(feat, churn_q)
show("Model AUC (tickets + tenure)", roc_auc_score(churn_q, model.predict_proba(feat)[:, 1]))
pred_after = model.predict_proba(feat.assign(tickets=0))[:, 1].mean()
show("Churn rate today", churn_q.mean())
show("MODEL predicts if tickets forced to 0", pred_after)
show("ACTUAL churn after the policy", rng.binomial(1, p_true).mean())
note("""The model 'promises' a big churn drop. Reality: no change - quality is
still bad, customers just can't complain. Feeding a predictive model a
'what if' it was never trained to answer is a classic, expensive mistake.
Use predictive models to RANK and TARGET; use experiments to decide WHAT
TO CHANGE.""")

banner("KEY TAKEAWAYS")
print("""  1. Correlation is enough to PREDICT; causation is needed to INTERVENE.
  2. Watch for confounders, reverse causation, selection bias, and spurious trends.
  3. Targeted treatments make naive before/after or treated/untreated comparisons lie.
  4. Randomized experiments are the gold standard; adjustments work only if you measured every confounder.
  5. Feature importance tells you what PREDICTS, not what to CHANGE.""")
