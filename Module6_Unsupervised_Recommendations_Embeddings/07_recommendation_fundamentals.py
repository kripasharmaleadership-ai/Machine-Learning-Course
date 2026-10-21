"""
LAB 07 - Recommendation-System Fundamentals
===========================================

THE PROBLEM
    800 users have each rated a handful of 400 movies (95%+ of the user x movie table is empty).
    For each user, which unseen movies should we show first?

FOUR FAMILIES OF RECOMMENDERS
    popularity        recommend what most people liked                    (no personalisation: the baseline)
    content-based     recommend items whose DESCRIPTION is similar to      (needs item features; works for
                      what the user liked: cosine similarity                new items; can't surprise)
    collaborative     'users who liked X also liked Y': similarity from    (needs no item features; fails
    filtering (CF)    the ratings themselves (item-item cosine)            for brand-new items/users)
    matrix            every user and every item becomes a short vector     (CF's strongest classic form;
    factorisation     (an EMBEDDING); rating ~ user . item                 the bridge to embeddings)

EVALUATION (offline): hide 3 movies each user loved; recommend 10 from everything unseen; did the hidden
    ones show up?  precision@10, recall@10, NDCG@10 (rank-aware), plus COVERAGE (how much of the catalogue
    gets recommended at all) and POPULARITY (how mainstream the recommendations are).

THE DATA (synthetic, offline): each user has a HIDDEN taste vector over 8 genres; ratings come from it.

LEVELS
    EASY   : The ratings matrix, the long tail, a popularity baseline, how offline evaluation works.
    MEDIUM : Content-based, item-item CF and matrix factorisation (ALS, from scratch), evaluated identically.
    HARD   : Cold start (new users, new items); do the learned embeddings recover hidden tastes?
             Popularity bias, coverage, and a hybrid.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from ml_lab_utils import GENRES, HAS_MPL, banner, make_ratings, note, plt, save_fig, show, step
from unsup_toolkit import ALSMatrixFactorization, leave_n_out_split, ndcg_at_k, precision_at_k, recall_at_k

ratings, movies, TRUE = make_ratings(n_users=800, n_items=400, seed=7)
N_U, N_I, K = 800, 400, 10
train, test = leave_n_out_split(ratings, n=3, min_rating=4.0)
R = np.zeros((N_U, N_I))
R[train.user, train.item] = train.rating
seen = R > 0
test_sets = test.groupby("user").item.apply(set).to_dict()
item_pop = np.bincount(train.item, minlength=N_I)
# only possible with synthetic data: each user's genuinely best 5% of UNSEEN movies, by true expected rating
TRUE_TOP = [np.flatnonzero((TRUE[u] >= np.quantile(TRUE[u][~seen[u]], 0.95)) & ~seen[u]) for u in range(N_U)]


def evaluate(scores: np.ndarray, name: str, users=None) -> dict:
    """Rank all UNSEEN items for each test user by `scores` (users x items); compare the top K with the hidden likes."""
    users = list(test_sets) if users is None else users
    s = np.where(seen, -np.inf, scores)
    top = np.argsort(-s, axis=1)[:, :K]
    p = [precision_at_k(list(top[u]), test_sets[u], K) for u in users]
    r = [recall_at_k(list(top[u]), test_sets[u], K) for u in users]
    n = [ndcg_at_k(list(top[u]), test_sets[u], K) for u in users]
    recs = top[users]
    taste = [np.mean(np.isin(top[u], TRUE_TOP[u])) for u in users]
    return {"precision@10": np.mean(p), "recall@10": np.mean(r), "NDCG@10": np.mean(n), "TRUE-taste precision@10": np.mean(taste),
            "catalogue coverage": len(np.unique(recs)) / N_I,
            "avg popularity of recs (ratings)": item_pop[recs].mean()}


# %% =========================================================================
# LEVEL 1 (EASY): The data, the long tail, the baseline
# ============================================================================
banner("The ratings matrix and a popularity baseline", "EASY")
show("Users x movies / ratings", f"{N_U} x {N_I} / {len(ratings):,}")
show("Share of the matrix that is empty", f"{1 - len(ratings) / (N_U * N_I):.1%}")
per_item = ratings.groupby("item").size().sort_values(ascending=False)
show("Ratings on the top 10% of movies", f"{per_item.head(N_I // 10).sum() / len(ratings):.0%}")
show("Movies with fewer than 5 ratings", f"{(per_item < 5).sum() + (N_I - len(per_item))} of {N_I}")
show("Held-out test likes / users evaluated", f"{len(test)} / {len(test_sets)}")
print(movies.head(4)[["title", "genres", "description"]].to_string())
note("""Most cells are empty, and ratings pile up on a few blockbusters (the LONG TAIL:
most movies are rarely rated). For evaluation, three movies each user rated 4+
are hidden; a recommender ranks all movies the user hasn't seen, and we check
whether the hidden ones land in its top 10.""")
results = {}
damped = (train.groupby("item").rating.sum().reindex(range(N_I), fill_value=0) + 10 * 3.0) / (item_pop + 10)
results["popularity: most rated"] = evaluate(np.tile(item_pop.astype(float), (N_U, 1)), "pop count")
results["popularity: best damped average rating"] = evaluate(np.tile(damped.to_numpy(), (N_U, 1)), "pop")
results["random"] = evaluate(np.random.default_rng(0).random((N_U, N_I)), "random")
results["TRUE expected ratings (knows taste, not exposure)"] = evaluate(TRUE, "oracle")
print(pd.DataFrame(results).T.round(3).to_string())
base_tab = pd.DataFrame(results).T
note(f"""Two non-personalised baselines show everyone the same 10 movies. Surprise:
'most rated' wins the offline test by far (NDCG {base_tab.iloc[0, 2]:.3f}), even beating a model that
knows every user's TRUE expected rating ({base_tab.iloc[3, 2]:.3f}). Why? The test hides movies users
actually RATED, and people rate what they were EXPOSED to: blockbusters. So
this lab adds a second lens, only possible with synthetic data: 'TRUE-taste
precision', the share of the top 10 that are among the user's genuinely best
5% of unseen movies. Through that lens 'most rated' scores {base_tab.iloc[0, 3]:.3f}, barely above
random ({base_tab.iloc[2, 3]:.3f}). Offline metrics measure exposure as much as taste.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Content-based, collaborative filtering, matrix factorisation
# ============================================================================
banner("Three ways to personalise", "MEDIUM")
step("2.1  Content-based: recommend movies whose description resembles what you liked")
tfidf = TfidfVectorizer().fit_transform(movies.description + " " + movies.genres.str.replace("|", " "))
item_vecs = tfidf.toarray()
item_vecs /= np.linalg.norm(item_vecs, axis=1, keepdims=True)
liked = np.where(seen, R - R.sum(1, keepdims=True) / np.maximum(seen.sum(1, keepdims=True), 1), 0)   # above / below own average
profiles = liked @ item_vecs                                            # a user = the sum of liked item vectors
content_scores = profiles @ item_vecs.T
results["content-based (TF-IDF profiles)"] = evaluate(content_scores, "content")

step("2.2  Item-item collaborative filtering: 'people who liked this also liked'")
centred = np.where(seen, R - np.where(seen, R, np.nan).mean(1, keepdims=True).__array__(), 0)
centred = np.nan_to_num(centred)
norms = np.linalg.norm(centred, axis=0) + 1e-9
item_sim = (centred.T @ centred) / np.outer(norms, norms)               # cosine between item rating columns
np.fill_diagonal(item_sim, 0)
cf_scores = centred @ item_sim
results["item-item CF (cosine)"] = evaluate(cf_scores, "cf")

step("2.3  Matrix factorisation by alternating least squares")
als = ALSMatrixFactorization(factors=8, reg=5.0, n_iter=15).fit(train, N_U, N_I)
show("Training RMSE by iteration (first, last)", f"{als.history_[0]:.3f} -> {als.history_[-1]:.3f}")
mf_scores = als.predict_all()
results["matrix factorisation (ALS, 8 factors)"] = evaluate(mf_scores, "mf")
res = pd.DataFrame(results).T
print(res.round(3).to_string())
note(f"""The two lenses disagree. Content-based is weak offline but finds the most
genuinely well-matched movies (TRUE-taste {res.loc['content-based (TF-IDF profiles)', 'TRUE-taste precision@10']:.3f}): it matches genres directly. Matrix
factorisation (8 numbers per user and movie, plus biases) is the best
PERSONALISED model offline (NDCG {res.loc['matrix factorisation (ALS, 8 factors)', 'NDCG@10']:.3f}) and good on taste ({res.loc['matrix factorisation (ALS, 8 factors)', 'TRUE-taste precision@10']:.3f}), partly because its
item biases learn popularity (its recommendations average {res.loc['matrix factorisation (ALS, 8 factors)', 'avg popularity of recs (ratings)']:.0f} ratings vs ~25).
Item-item CF struggles on data this sparse. Choosing a recommender on the
offline number alone would have picked 'most rated'.""")

# %% =========================================================================
# LEVEL 3 (HARD): Cold start, embeddings, bias and a hybrid
# ============================================================================
banner("Cold start, learned embeddings, popularity bias, a hybrid", "HARD")
step("3.1  Cold start: brand-new movies have no ratings")
new_items = np.random.default_rng(1).choice(N_I, 40, replace=False)    # pretend 40 movies were added yesterday
R_cold, seen_cold = R.copy(), seen.copy()
R_cold[:, new_items], seen_cold[:, new_items] = 0, False
train_cold = train[~train.item.isin(new_items)]
targets = {u: s & set(new_items) for u, s in test_sets.items() if s & set(new_items)}
liked_c = np.where(seen_cold, R_cold - R_cold.sum(1, keepdims=True) / np.maximum(seen_cold.sum(1, keepdims=True), 1), 0)
content_cold = (liked_c @ item_vecs) @ item_vecs[new_items].T
als_c = ALSMatrixFactorization(factors=8, reg=5.0, n_iter=10).fit(train_cold, N_U, N_I)
mf_cold = als_c.predict_all()[:, new_items]                             # new items' vectors were never updated
hits = {}
for name, sc in {"content-based": content_cold, "matrix factorisation": mf_cold}.items():
    top = np.argsort(-sc, axis=1)[:, :5]
    hits[name] = np.mean([len(set(new_items[top[u]]) & targets[u]) > 0 for u in targets])
show("Users whose hidden like is a NEW movie", len(targets))
print(pd.Series(hits, name="share whose new-movie like is in their top 5 of the 40 new movies").round(3).to_string())
note(f"""A movie nobody has rated has no collaborative signal: matrix factorisation ranks
the 40 new movies essentially at random (picking 5 of 40 at random finds the
liked one 12.5% of the time; MF: {hits['matrix factorisation']:.1%}). Content-based recommendation only
needs the description, so it has SOME signal on day one ({hits['content-based']:.1%}), modest here
because descriptions are short tag lists. That is why production systems are
HYBRIDS: content / metadata for the cold start, collaborative signals once
interactions accumulate.""")

step("3.2  Do the learned user embeddings recover the hidden tastes?")
taste_items = np.array([[g in genres.split("|") for g in GENRES] for genres in movies.genres], dtype=float)
true_pref = (TRUE - TRUE.mean(1, keepdims=True)) @ (taste_items / taste_items.sum(0)) # each user's true liking of each genre
r2 = {g: cross_val_score(LinearRegression(), als.P_, true_pref[:, j], cv=5, scoring="r2").mean() for j, g in enumerate(GENRES)}
print(pd.Series(r2, name="R^2: genre preference predicted from the 8 learned numbers per user").round(3).to_string())
note(f"""Nobody told the model what genres are. Yet the 8 numbers it learned for each user
partly predict that user's true preference for each genre (mean R^2 {np.mean(list(r2.values())):.2f}, up to
{max(r2.values()):.2f} for {max(r2, key=r2.get)}), from only ~15 ratings per user. Matrix
factorisation discovered the hidden taste dimensions from ratings alone: the
user vectors are EMBEDDINGS of taste, and similar users have nearby vectors.
The same idea, applied to words, gives word embeddings (Lab 08).""")

step("3.3  Popularity bias, and a simple hybrid")
z = lambda m: (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)
hybrid = 0.7 * z(mf_scores) + 0.3 * z(content_scores)
results["hybrid: 0.7 MF + 0.3 content"] = evaluate(hybrid, "hybrid")
pop_pen = mf_scores - 0.15 * np.log1p(item_pop)[None, :]
results["MF with a popularity penalty"] = evaluate(pop_pen, "pen")
final = pd.DataFrame(results).T
print(final.round(3).to_string())
mf_row = final.loc["matrix factorisation (ALS, 8 factors)"]
pen_row = final.loc["MF with a popularity penalty"]
hyb_row = final.loc["hybrid: 0.7 MF + 0.3 content"]
note(f"""The hybrid (70% matrix factorisation, 30% content) is the best personalised
model on BOTH lenses (NDCG {hyb_row['NDCG@10']:.3f}, TRUE-taste {hyb_row['TRUE-taste precision@10']:.3f}) and covers more of the catalogue
({mf_row['catalogue coverage']:.0%} -> {hyb_row['catalogue coverage']:.0%}). The popularity penalty halves how mainstream the recommendations
are ({mf_row['avg popularity of recs (ratings)']:.0f} -> {pen_row['avg popularity of recs (ratings)']:.0f} ratings) and keeps almost all of the true-taste precision ({pen_row['TRUE-taste precision@10']:.3f}),
yet the offline NDCG halves: an offline-only evaluation would wrongly reject
it. In real life nobody knows the true taste, so decisions like this are
settled with online A/B tests that measure what users actually do next.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(np.arange(1, N_I + 1), np.sort(np.bincount(ratings.item, minlength=N_I))[::-1])
    axes[0].set(xlabel="movies ranked by number of ratings", ylabel="ratings", yscale="log", title="The long tail")
    f = final.drop(index=["random"])
    f = f.drop(index=["TRUE expected ratings (knows taste, not exposure)"])
    axes[1].scatter(f["NDCG@10"], f["TRUE-taste precision@10"], s=60)
    for n_, r_ in f.iterrows():
        axes[1].annotate(n_.split(" (")[0], (r_["NDCG@10"], r_["TRUE-taste precision@10"]), fontsize=7, xytext=(4, 3),
                         textcoords="offset points")
    axes[1].set(xlabel="offline NDCG@10 (hidden ratings)", ylabel="TRUE-taste precision@10",
                title="Two lenses disagree: offline rewards exposure")
    axes[2].bar(list(r2), list(r2.values()), color="#4c78a8")
    axes[2].set(ylabel="R^2", title="Learned user vectors predict hidden genre tastes")
    axes[2].tick_params(axis="x", rotation=45)
    save_fig(fig, "07_recommendations")

banner("KEY TAKEAWAYS")
print("""  1. Recommenders rank unseen items per user; evaluate offline by hiding liked items (precision/recall/NDCG@k).
  2. Offline metrics reward EXPOSURE: 'most rated' wins offline yet barely matches true taste. Report coverage too.
  3. Content-based uses item descriptions (works for new items); CF uses who-rated-what (needs history).
  4. Matrix factorisation learns a vector per user and item: embeddings that recover hidden tastes.
  5. Production systems are hybrids that balance accuracy, coverage, cold start and popularity bias.""")
