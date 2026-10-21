"""
LAB 06 - Anomaly Detection
==========================

THE IDEA
    Find the rows that don't look like the rest, WITHOUT labels saying which ones are bad.
        statistical rules   z-score > 3, outside 1.5 x IQR: one feature at a time, easy to explain
        Isolation Forest    random trees isolate odd points in very few splits (short paths = anomalous)
        Local Outlier Factor a point much less dense than its neighbours is an outlier
        DBSCAN noise        points in no dense region (Lab 04)

    An anomaly score ranks rows by 'how unusual'. A THRESHOLD (or an alert budget: 'review the top 200
    per month') turns the ranking into decisions, exactly as in Module 4.

THE DATA: 8,000 employee expense claims with planted problems (`_anomaly`, never used for training):
    inflated  : amount far above normal for the category
    weekend   : a large claim submitted on a weekend night
    split     : repeated claims just under the $500 approval limit
    mismatch  : a junior claiming a flight in a month with no travel

LEVELS
    EASY   : z-score and IQR rules; what they catch, kind by kind.
    MEDIUM : Isolation Forest from scratch (= scikit-learn's idea); Local Outlier Factor; who catches what.
    HARD   : Domain features beat clever algorithms; alert budgets; from unsupervised to supervised
             (auditors label the top alerts, a classifier learns from them).
"""
# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import EXPENSE_CATEGORIES, HAS_MPL, banner, make_expense_claims, note, plt, save_fig, show, step
from unsup_toolkit import isolation_scores

claims = make_expense_claims(n=8000, seed=6)
y = (claims._anomaly != "normal").to_numpy()
KINDS = ["inflated", "weekend", "split", "mismatch"]
BUDGET = int(0.05 * len(claims))                                       # auditors can review 5% of claims


def by_kind(score, k=BUDGET) -> dict:
    """Recall per anomaly kind when the top-k scores are reviewed, plus precision of the reviewed set."""
    flagged = np.zeros(len(score), bool)
    flagged[np.argsort(-np.asarray(score))[:k]] = True
    out = {kind: flagged[(claims._anomaly == kind).to_numpy()].mean() for kind in KINDS}
    out["precision of the review list"] = y[flagged].mean()
    out["ROC-AUC"] = roc_auc_score(y, score)
    return out


def design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(df[["category"]], dtype=float)
    return X.assign(log_amount=np.log(df.amount), level=df.employee_level, weekday=df.weekday, hour=df.hour,
                    travel_days=df.travel_days_this_month)


X = StandardScaler().fit_transform(design(claims))
show("Claims / planted anomalies", f"{len(claims):,} / {y.sum()} ({y.mean():.1%})")
show("Review budget (5% of claims)", BUDGET)

# %% =========================================================================
# LEVEL 1 (EASY): Statistical rules
# ============================================================================
banner("z-scores and the IQR rule", "EASY")
la = np.log(claims.amount)
grp = claims.groupby("category")
z = (la - grp["amount"].transform(lambda a: np.log(a).mean())) / grp["amount"].transform(lambda a: np.log(a).std())
q1, q3 = grp["amount"].transform(lambda a: a.quantile(0.25)), grp["amount"].transform(lambda a: a.quantile(0.75))
iqr_flag = (claims.amount > q3 + 1.5 * (q3 - q1)).to_numpy()
z_flag = (np.abs(z) > 3).to_numpy()
rows = {}
for name, flag in {"|z| > 3 (log amount, per category)": z_flag, "amount > Q3 + 1.5 IQR (per category)": iqr_flag}.items():
    rows[name] = {"flagged": int(flag.sum()), **{k: flag[(claims._anomaly == k).to_numpy()].mean() for k in KINDS},
                  "precision": y[flag].mean() if flag.any() else np.nan}
rule_tab = pd.DataFrame(rows).T.astype({"flagged": int})
print(rule_tab.round(3).to_string())
note(f"""A z-score asks 'how many standard deviations from this category's typical
amount?'; the IQR rule flags amounts beyond the upper fence. Both work PER
CATEGORY (a $900 flight is normal, a $900 lunch is not). They catch most
inflated claims and, being expensive, many weekend and junior-flight claims. But
the training courses booked at $470-499 are perfectly normal amounts for
training: caught {rule_tab.loc[rule_tab.index[1], 'split']:.0%}. And the strict |z| > 3 rule is precise but finds few
problems; the looser IQR rule finds more at {rule_tab.iloc[1, -1]:.0%} precision. One column at a time
only sees what is extreme in that column.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Isolation Forest and LOF
# ============================================================================
banner("Isolation Forest (from scratch) and Local Outlier Factor", "MEDIUM")
step("2.1  How an isolation tree works: the most typical claim vs an inflated one")
rng = np.random.default_rng(0)
ref = -IsolationForest(n_estimators=300, random_state=0).fit(X).score_samples(X)
normal_i = int(np.argmin(np.where(~y, ref, np.inf)))                       # the most typical claim
odd_i = int(np.argmax(np.where((claims._anomaly == "inflated").to_numpy(), ref, -np.inf)))   # the most unusual inflated one
depths = {"normal claim": [], "inflated claim": []}
for _ in range(200):
    for label, i in [("normal claim", normal_i), ("inflated claim", odd_i)]:
        data, d = X[rng.choice(len(X), 256, replace=False)], 0
        while len(data) > 1 and d < 30:
            j = rng.integers(X.shape[1])
            lo, hi = data[:, j].min(), data[:, j].max()
            if lo == hi:
                break
            t = rng.uniform(lo, hi)
            data = data[data[:, j] < t] if X[i, j] < t else data[data[:, j] >= t]
            d += 1
        depths[label].append(d)
show("Average random splits to isolate: normal / inflated", f"{np.mean(depths['normal claim']):.1f} / {np.mean(depths['inflated claim']):.1f}")
note("""Pick a random feature and a random cut, keep the side the claim falls on,
repeat until it is alone. An unusual claim sits in an empty region, so a few
random cuts isolate it; a typical claim is surrounded by similar ones and
needs many. The average path length over many random trees IS the anomaly
score. No distances, no density estimates: fast on big data.""")

step("2.2  Three detectors, the same review budget")
sub = np.random.default_rng(1).choice(len(X), 2500, replace=False)        # the scratch version is slow: score a sample
scratch = isolation_scores(X[sub], n_trees=40)
iso = IsolationForest(n_estimators=300, random_state=0).fit(X)
iso_score = -iso.score_samples(X)
show("Scratch vs scikit-learn Isolation Forest (rank correlation, 2,500 claims)",
     f"{pd.Series(scratch).corr(pd.Series(iso_score[sub]), method='spearman'):.3f}")
lof_score = -LocalOutlierFactor(n_neighbors=30).fit(X).negative_outlier_factor_
detectors = {"|z| of log amount (per category)": np.abs(z).to_numpy(), "Isolation Forest": iso_score, "Local Outlier Factor": lof_score}
det_tab = pd.DataFrame({name: by_kind(sc) for name, sc in detectors.items()}).T
print(det_tab.round(3).to_string())
note(f"""Same budget, three ideas of 'unusual', three different catches. Isolation Forest
has the best overall ranking (ROC-AUC {det_tab.loc['Isolation Forest', 'ROC-AUC']:.2f}) and catches odd COMBINATIONS
(weekend nights, juniors flying), but only {det_tab.loc['Isolation Forest', 'inflated']:.0%} of inflated amounts: its random
splits are spread over 11 columns, and an extreme value in ONE column is easy
to miss. LOF and the per-category z-score catch the inflated ones. None catches more than ~a quarter
of the split training claims: they aren't unusual in any column. An anomaly
detector only sees what is unusual in the features you give it.""")

# %% =========================================================================
# LEVEL 3 (HARD): Domain features, budgets, and labels
# ============================================================================
banner("Domain knowledge, alert budgets, and the path to supervision", "HARD")
step("3.1  Features that encode what 'suspicious' means")
eng = design(claims).assign(
    amount_vs_category_median=claims.amount / claims.groupby("category").amount.transform("median"),
    just_under_limit=claims.amount.between(450, 499.99).astype(float),
    weekend_night=((claims.weekday >= 5) & ((claims.hour <= 4) | (claims.hour >= 23))).astype(float),
    flight_without_travel=((claims.category == "flight") & (claims.travel_days_this_month == 0)).astype(float),
    junior_expensive=((claims.employee_level == 1) & (claims.amount > 500)).astype(float))
Xe = StandardScaler().fit_transform(eng)
iso_eng = -IsolationForest(n_estimators=300, random_state=0).fit(Xe).score_samples(Xe)
eng_tab = pd.DataFrame({"Isolation Forest, raw features": by_kind(iso_score),
                        "Isolation Forest + 5 domain features": by_kind(iso_eng)}).T
print(eng_tab.round(3).to_string())
note(f"""Five features written with an auditor (ratio to the category's typical amount,
'just under the approval limit', weekend night, flight without travel, junior
with a large claim) turn hidden patterns into extreme values: the SAME
algorithm goes from ROC-AUC {eng_tab.iloc[0, -1]:.2f} to {eng_tab.iloc[1, -1]:.2f}, and the review list from {eng_tab.iloc[0, -2]:.0%} to {eng_tab.iloc[1, -2]:.0%} real
problems. Split claims go from {eng_tab.iloc[0, 2]:.0%} to {eng_tab.iloc[1, 2]:.0%}: their pattern lives in a THRESHOLD ($500) the
data alone doesn't know about. Domain knowledge beats algorithm choice here.""")

step("3.2  The alert budget is the real decision")
rows = {}
for budget in [50, 100, 200, 400, 800]:
    top = np.argsort(-iso_eng)[:budget]
    rows[budget] = {"precision (share of reviews that are real problems)": y[top].mean(),
                    "recall (share of all problems found)": y[top].sum() / y.sum()}
budget_tab = pd.DataFrame(rows).T
budget_tab.index.name = "claims reviewed per period"
print(budget_tab.round(3).to_string())
note("""IsolationForest's `contamination` parameter only sets where the threshold sits;
the scores don't change. The real question is the review capacity: a small
list is precise but finds few problems, a long list finds most but wastes
auditor time on oddities that are legitimate (an expensive course, a senior's
late-night flight). Choose the budget with the team that does the reviews,
as in Module 4.""")

step("3.3  From unsupervised to supervised: learn from the auditors' verdicts")
reviewed = np.argsort(-iso_eng)[:400]                                   # month 1: auditors review the top 400 alerts
labels = y[reviewed]                                                    # ... and record which were real problems
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(eng.iloc[reviewed], labels)
nxt = make_expense_claims(n=8000, seed=60)                              # month 2: new claims
y2 = (nxt._anomaly != "normal").to_numpy()
eng2 = design(nxt).reindex(columns=design(claims).columns, fill_value=0).assign(
    amount_vs_category_median=nxt.amount / nxt.groupby("category").amount.transform("median"),
    just_under_limit=nxt.amount.between(450, 499.99).astype(float),
    weekend_night=((nxt.weekday >= 5) & ((nxt.hour <= 4) | (nxt.hour >= 23))).astype(float),
    flight_without_travel=((nxt.category == "flight") & (nxt.travel_days_this_month == 0)).astype(float),
    junior_expensive=((nxt.employee_level == 1) & (nxt.amount > 500)).astype(float))
iso2 = -IsolationForest(n_estimators=300, random_state=0).fit(StandardScaler().fit_transform(eng2)).score_samples(
    StandardScaler().fit_transform(eng2))
sup2 = clf.predict_proba(eng2)[:, 1]
loop = pd.DataFrame({name: {"precision in the top 400": y2[np.argsort(-s)[:400]].mean(), "average precision": average_precision_score(y2, s)}
                     for name, s in {"month 2: Isolation Forest only": iso2,
                                     "month 2: classifier trained on month-1 audits": sup2}.items()}).T
print(loop.round(3).to_string())
show("Labels the auditors produced in month 1 (real problems / false alarms)", f"{labels.sum()} / {(~labels).sum()}")
note("""An anomaly score is a way to FIND candidates, not a verdict. Every review
produces a label. After one month, a classifier trained on just 400 audited
claims (problems AND false alarms) learns what the auditors care about and
ranks month 2 better than the unsupervised score. Unsupervised to start,
supervised as labels accumulate, with anomaly detection kept running to
catch NEW kinds of problems the classifier has never seen.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].hist(depths["normal claim"], bins=range(0, 16), alpha=0.6, label="normal claim")
    axes[0].hist(depths["inflated claim"], bins=range(0, 16), alpha=0.6, label="inflated claim")
    axes[0].set(xlabel="random splits needed to isolate", title="Isolation: anomalies have short paths")
    axes[0].legend()
    et = eng_tab[KINDS].T
    et.plot.bar(ax=axes[1], rot=0)
    axes[1].set(ylabel=f"recall in the top {BUDGET}", title="Domain features reveal hidden kinds")
    axes[2].plot(budget_tab.index, budget_tab.iloc[:, 0], "o-", label="precision")
    axes[2].plot(budget_tab.index, budget_tab.iloc[:, 1], "s-", label="recall")
    axes[2].set(xscale="log", xlabel="claims reviewed", title="The alert budget trade-off")
    axes[2].legend()
    save_fig(fig, "06_anomaly_detection")

banner("KEY TAKEAWAYS")
print("""  1. Anomaly detection ranks rows by unusualness without labels; a budget or threshold turns it into action.
  2. Rules (z-score, IQR) catch extremes on one feature; Isolation Forest and LOF catch odd combinations.
  3. Detectors only see what the features express: domain features reveal patterns like 'just under the limit'.
  4. The review budget sets the precision-recall trade-off (Module 4 again).
  5. Use anomalies to collect labels, then add a supervised model; keep the detector for new patterns.""")
