"""
LAB 06 - Regression, Classification, Clustering and Recommendation
===================================================================

FOUR QUESTIONS, FOUR TASK TYPES
    REGRESSION      "HOW MUCH / HOW MANY?"     -> a number         (next month's spend: $142)
    CLASSIFICATION  "WHICH ONE / YES or NO?"   -> a class + prob   (will churn: yes, 73%)
    CLUSTERING      "WHAT GROUPS EXIST?"        -> a group id       (segment 2 = "bargain hunters")
    RECOMMENDATION  "WHAT WILL THEY LIKE?"      -> a ranked list    (top-10 products for user 17)

    Each one has its OWN metrics. Using the wrong metric is a top-5 reason
    ML projects fail (see Lab 13).

LEVELS
    EASY   : Four tiny, readable examples - one per task type.
    MEDIUM : ONE e-commerce business, framed four ways, each with the right metric
             and an honest baseline. Choosing the number of clusters with data.
    HARD   : A real recommender pipeline: implicit feedback, leave-one-out
             evaluation, popularity vs item-similarity vs matrix factorization,
             measured with HitRate@10 - the way recommender teams actually work.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score, root_mean_squared_error, silhouette_score,
                             adjusted_rand_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (HAS_MPL, RANDOM_STATE, banner, make_ecommerce_customers, note,
                          plt, save_fig, show, step)

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): one tiny example per task type
# ============================================================================
banner("Four task types in four mini-examples", "EASY")

step("1) REGRESSION - predict disease progression (a number) for diabetes patients")
Xr, yr = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=RANDOM_STATE)
reg = LinearRegression().fit(Xr_tr, yr_tr)
show("Prediction for first test patient", f"{reg.predict(Xr_te[:1])[0]:.0f} (actual {yr_te[0]:.0f})")
show("MAE (avg error, same units as target)", mean_absolute_error(yr_te, reg.predict(Xr_te)))

step("2) CLASSIFICATION - malignant or benign tumour (a class)")
Xc, yc = load_breast_cancer(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25, stratify=yc, random_state=RANDOM_STATE)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xc_tr, yc_tr)
show("Prediction for first test patient", f"class {clf.predict(Xc_te[:1])[0]} "
     f"(P(benign)={clf.predict_proba(Xc_te[:1])[0, 1]:.2f}, actual {yc_te[0]})")
show("Accuracy", accuracy_score(yc_te, clf.predict(Xc_te)))

step("3) CLUSTERING - group shoppers with NO labels")
shoppers = pd.DataFrame({"visits_per_month": [1, 2, 1, 12, 15, 14, 3, 2, 13],
                         "avg_basket_$": [200, 180, 220, 20, 25, 30, 210, 190, 22]})
shoppers["cluster"] = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit_predict(
    StandardScaler().fit_transform(shoppers))
print(shoppers.to_string(index=False))
note("Two groups emerge: 'rare big-basket' vs 'frequent small-basket'. Nobody labelled them.")

step("4) RECOMMENDATION - 'people who liked X also liked Y'")
ratings = pd.DataFrame(
    [[5, 4, 0, 0, 1], [4, 5, 0, 1, 0], [0, 0, 5, 4, 0], [0, 1, 4, 5, 0], [5, 5, 0, 0, 0]],
    index=["Asha", "Ben", "Chen", "Dina", "Eli"],
    columns=["Inception", "Interstellar", "Notebook", "Titanic", "Toy Story"],
)  # 0 = not watched
print(ratings)
item_sim = pd.DataFrame(cosine_similarity(ratings.T), index=ratings.columns, columns=ratings.columns)
print("\n  Most similar to 'Inception':", item_sim["Inception"].drop("Inception").idxmax())
scores = ratings.loc["Eli"] @ item_sim               # weight similar items by what Eli liked
scores[ratings.loc["Eli"] > 0] = -np.inf             # never recommend what was already watched
show("Recommend to Eli", scores.idxmax())
note("""Recommenders output a RANKED LIST, not a single answer. Here: items rated
similarly by the same people are 'similar' (item-based collaborative filtering).""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): One business, four framings, correct metrics + baselines
# ============================================================================
banner("One e-commerce company, four ML framings", "MEDIUM")
shop = make_ecommerce_customers()
features = ["visits_90d", "avg_basket_value", "discount_share", "days_since_last_order",
            "returns_rate", "spend_last_90d"]
print(shop.drop(columns="_true_segment").head())
train, test = train_test_split(shop, test_size=0.25, random_state=RANDOM_STATE)

step("Framing A - REGRESSION: 'How much will each customer spend next month?'")
base_r = DummyRegressor(strategy="mean").fit(train[features], train.next_month_spend)
gbr = GradientBoostingRegressor(random_state=RANDOM_STATE).fit(train[features], train.next_month_spend)
for name, m in [("baseline (predict average)", base_r), ("gradient boosting", gbr)]:
    p = m.predict(test[features])
    print(f"  {name:<28} MAE=${mean_absolute_error(test.next_month_spend, p):6.2f}  "
          f"RMSE=${root_mean_squared_error(test.next_month_spend, p):6.2f}  R2={r2_score(test.next_month_spend, p):.3f}")
note("""MAE = average dollar error (easy to explain to finance). RMSE punishes big
misses more. R2 = share of variation explained (0 = no better than average).""")

step("Framing B - CLASSIFICATION: 'Will this customer churn?'")
show("Churn rate", f"{shop.will_churn.mean():.1%}")
base_c = DummyClassifier(strategy="most_frequent").fit(train[features], train.will_churn)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE)
rf.fit(train[features], train.will_churn)
for name, m in [("baseline (majority class)", base_c), ("random forest", rf)]:
    pred, proba = m.predict(test[features]), m.predict_proba(test[features])[:, 1]
    print(f"  {name:<28} acc={accuracy_score(test.will_churn, pred):.3f}  "
          f"precision={precision_score(test.will_churn, pred, zero_division=0):.3f}  "
          f"recall={recall_score(test.will_churn, pred):.3f}  AUC={roc_auc_score(test.will_churn, proba):.3f}")
note("""Precision = of customers we flag, how many really churn (wasted offers).
Recall = of customers who churn, how many we catch (lost revenue). The
baseline's accuracy looks OK but it catches NOBODY - accuracy alone lies.""")

step("Framing C - CLUSTERING: 'What kinds of customers do we have?'")
X_seg = StandardScaler().fit_transform(shop[features])
print("  Choosing k with the silhouette score (higher = better separated groups):")
sil = {}
for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_seg)
    sil[k] = silhouette_score(X_seg, labels, sample_size=2000, random_state=RANDOM_STATE)
    print(f"    k={k}: silhouette={sil[k]:.3f}")
best_k = max(sil, key=sil.get)
shop["segment"] = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_seg)
profile = shop.groupby("segment")[features + ["next_month_spend"]].mean().round(2)
profile["customers"] = shop.segment.value_counts().sort_index()
print(f"\n  Segment profiles (k={best_k}):")
print(profile.to_string())
show("Match with hidden true segments (ARI)", adjusted_rand_score(shop._true_segment, shop.segment))
note("""Clustering output is a starting point for HUMANS: marketing names the
segments ('bargain hunters' = high discount_share, 'big spenders' = high
basket). No accuracy exists for clustering; you judge separation
(silhouette), stability, and above all business usefulness.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in sorted(shop.segment.unique()):
        part = shop[shop.segment == s]
        ax.scatter(part.discount_share, part.avg_basket_value, s=8, alpha=0.5, label=f"segment {s}")
    ax.set(xlabel="share of orders with discount", ylabel="average basket ($)",
           title="Customer segments found by KMeans")
    ax.legend()
    save_fig(fig, "06_medium_segments")

step("Framing D - RECOMMENDATION: 'What should we show each customer?'  -> see HARD level")

# %% =========================================================================
# LEVEL 3 (HARD): A recommender evaluated like a real team does it
# ============================================================================
banner("Recommender systems: popularity vs item-kNN vs matrix factorization", "HARD")
note("""Implicit feedback: we only know WHO BOUGHT WHAT (no star ratings) - the
normal situation in e-commerce. We simulate 800 users x 300 products, where
each user has hidden tastes over 6 product categories.""")

n_users, n_items, n_genres = 800, 300, 6
item_genre = rng.integers(n_genres, size=n_items)
item_popularity = rng.pareto(1.5, n_items) + 1            # a few blockbuster products
user_taste = rng.dirichlet(np.full(n_genres, 0.3), size=n_users)  # each user likes 1-2 genres
R = np.zeros((n_users, n_items))
for u in range(n_users):
    weights = user_taste[u, item_genre] * item_popularity
    bought = rng.choice(n_items, size=rng.integers(8, 25), replace=False, p=weights / weights.sum())
    R[u, bought] = 1
show("Users x items", R.shape)
show("Density (share of cells filled)", f"{R.mean():.2%}")

step("Leave-one-out split: hide ONE purchase per user, try to recover it")
R_train = R.copy()
held_out = np.zeros(n_users, dtype=int)
for u in range(n_users):
    held_out[u] = rng.choice(np.flatnonzero(R[u]))
    R_train[u, held_out[u]] = 0


def hit_rate_at_k(score_matrix: np.ndarray, k: int = 10) -> float:
    """Share of users whose hidden item appears in their top-k recommendations."""
    scores = score_matrix.copy()
    scores[R_train > 0] = -np.inf                      # never recommend already-bought items
    top_k = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([held_out[u] in top_k[u] for u in range(n_users)]))


results = {}
# 1. Random - the floor
results["random"] = hit_rate_at_k(rng.random(R.shape))
# 2. Popularity - recommend best-sellers to everyone (a STRONG baseline, often hard to beat)
results["popularity"] = hit_rate_at_k(np.tile(R_train.sum(0), (n_users, 1)))
# 3. Item-kNN - items bought by the same people are similar
sim = cosine_similarity(R_train.T)
np.fill_diagonal(sim, 0)
results["item-kNN (cosine)"] = hit_rate_at_k(R_train @ sim)
# 4. Matrix factorization - compress users & items into k hidden 'taste' factors
for n_factors in [5, 20, 50]:
    svd = TruncatedSVD(n_components=n_factors, random_state=RANDOM_STATE)
    user_f = svd.fit_transform(R_train)
    results[f"matrix factorization (k={n_factors})"] = hit_rate_at_k(user_f @ svd.components_)

print(pd.Series(results, name="HitRate@10").round(3).to_string())
note("""Read this like a principal engineer:
 - Random ~3%: the floor (10 of ~290 candidate items).
 - Popularity is a surprisingly strong baseline - ALWAYS report it.
 - Personalized methods win because they exploit each user's taste.
 - More factors is not always better: too few underfit, too many start to
   memorize noise. The number of factors is a hyperparameter you tune on
   validation data (Lab 07), never on the final test set.""", keep_lines=True)

step("Show actual recommendations for one user (explainability)")
u = 7
svd = TruncatedSVD(n_components=5, random_state=RANDOM_STATE)   # best k from above
scores_u = (svd.fit_transform(R_train) @ svd.components_)[u]
scores_u[R_train[u] > 0] = -np.inf
top5 = np.argsort(-scores_u)[:5]
show("User's taste (genre weights)", np.round(user_taste[u], 2).tolist())
show("Genres of items they bought", np.bincount(item_genre[R_train[u] > 0], minlength=n_genres).tolist())
show("Genres of top-5 recommendations", item_genre[top5].tolist())
show("Hidden item / its genre", f"{held_out[u]} / {item_genre[held_out[u]]}")
show("Hidden item in top-10?", held_out[u] in np.argsort(-scores_u)[:10])
note("""Recommendations follow the user's strongest tastes (genres 3 and 4). Even when
the exact hidden item is missed, the list is sensible - which is why teams
also track coverage, diversity and online A/B results, not one offline
number. A HitRate@10 of ~0.33 means 2 of 3 users DON'T see their item.""")

banner("KEY TAKEAWAYS")
print(pd.DataFrame([
    ["Regression", "A number", "MAE, RMSE, R2", "Predict the mean"],
    ["Classification", "Class + probability", "Precision, recall, ROC-AUC", "Majority class"],
    ["Clustering", "Group id", "Silhouette + business review", "n/a (human judgement)"],
    ["Recommendation", "Ranked list", "HitRate@K, Precision@K, NDCG", "Most popular items"],
], columns=["Task", "Output", "Metrics", "Baseline to beat"]).to_string(index=False))
