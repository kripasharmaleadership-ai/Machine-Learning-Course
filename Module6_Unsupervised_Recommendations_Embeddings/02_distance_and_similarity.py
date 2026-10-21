"""
LAB 02 - Distance and Similarity
================================

EVERY METHOD IN THIS MODULE STANDS ON ONE QUESTION: "how alike are these two things?"
    Euclidean   sqrt(sum (a_i - b_i)^2)      straight-line distance; sensitive to scale and to SIZE
    Manhattan   sum |a_i - b_i|              city-block distance; less dominated by one big difference
    cosine      a.b / (|a| |b|)              the ANGLE between vectors: same direction = 1, ignores length
    Jaccard     |A n B| / |A u B|            overlap of two SETS (tags, purchased items, words)
    Hamming     number of positions that differ (binary vectors, codes)

    Distance (small = alike) and similarity (large = alike) are two views of the same idea.

LEVELS
    EASY   : Every measure by hand for three customers and two baskets; scikit-learn agrees.
    MEDIUM : The measure changes the answer: scaling, size vs mix (Euclidean vs cosine), text length.
    HARD   : The curse of dimensionality; correlated features count twice (Mahalanobis); similarity
             of LEARNED vectors (a preview of embeddings).
"""
# %%
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import HAS_MPL, SPEND_COLS, banner, make_customers, make_text_corpus, note, plt, save_fig, show, step
from unsup_toolkit import PPMIEmbeddings, cosine_sim, euclidean, jaccard, manhattan

# %% =========================================================================
# LEVEL 1 (EASY): Every measure by hand
# ============================================================================
banner("Five ways to say 'how alike'", "EASY")
three = pd.DataFrame({"groceries": [400, 420, 120], "electronics": [60, 80, 25], "travel": [50, 40, 12]},
                     index=["Ana", "Ben", "Cleo"])
print(three.to_string())
a, b, c = three.to_numpy(float)
rows = {}
for name, x, y in [("Ana vs Ben", a, b), ("Ana vs Cleo", a, c), ("Ben vs Cleo", b, c)]:
    rows[name] = {"Euclidean": euclidean(x, y), "Manhattan": manhattan(x, y), "cosine similarity": cosine_sim(x, y)}
by_hand = pd.DataFrame(rows).T
print(by_hand.round(3).to_string())
show("scikit-learn agrees (Euclidean, Manhattan, cosine for Ana vs Ben)",
     f"{euclidean_distances([a], [b])[0, 0]:.3f}, {manhattan_distances([a], [b])[0, 0]:.3f}, {cosine_similarity([a], [b])[0, 0]:.3f}")
note("""Ana and Ben spend similar amounts, so they are close in Euclidean and Manhattan
distance. Cleo spends far less, so she is far from both in distance, yet her
cosine similarity with Ana is close to 1: she spends in the SAME PROPORTIONS
(mostly groceries), only less. Distance measures how much; cosine measures
the direction (the mix). Which one is right depends on the question.""")

step("1.1  Sets and binary vectors: Jaccard and Hamming")
baskets = {"Ana": {"milk", "bread", "coffee", "apples"}, "Ben": {"milk", "bread", "beer"}, "Cleo": {"coffee", "apples", "tea"}}
for x, y in [("Ana", "Ben"), ("Ana", "Cleo"), ("Ben", "Cleo")]:
    show(f"Jaccard({x}, {y})", f"{jaccard(baskets[x], baskets[y]):.3f}  shared: {sorted(baskets[x] & baskets[y])}")
flags_a, flags_b = np.array([1, 0, 1, 1, 0, 0]), np.array([1, 1, 1, 0, 0, 0])
show("Hamming distance of two yes/no profiles", f"{int((flags_a != flags_b).sum())} of 6 positions differ")
note("""When the data is 'which items / tags / words are present', Jaccard (shared
items divided by all items) is the natural similarity: it ignores the endless
list of items neither person bought. It is the classic measure behind
'customers who bought this also bought' and near-duplicate detection.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): The measure changes the answer
# ============================================================================
banner("Same data, different 'nearest neighbour'", "MEDIUM")
customers = make_customers(n=2000, seed=2)
FEATS = ["age", "income_k"] + SPEND_COLS + ["visits_per_month", "online_share"]
X_raw = customers[FEATS].to_numpy(float)
X_std = StandardScaler().fit_transform(np.log1p(customers[FEATS]))

step("2.1  Scaling: who is the nearest neighbour of customer 0?")
d_raw = cdist(X_raw[:1], X_raw)[0]
d_std = cdist(X_std[:1], X_std)[0]
nn_raw, nn_std = np.argsort(d_raw)[1], np.argsort(d_std)[1]
print(customers.loc[[0, nn_raw, nn_std], FEATS + ["_segment"]].assign(
    role=["customer 0", "nearest (raw units)", "nearest (log + standardised)"]).set_index("role").round(1).to_string())
spread = pd.Series(X_raw.std(axis=0), index=FEATS).round(1)
note(f"""In raw units each feature contributes in its own unit: a $100 difference in
spend counts 100x more than a difference of 1 in visits, and 'online_share' (0-1)
barely counts at all. The raw nearest neighbour is the customer with similar
DOLLAR amounts; after log + standardising, every feature has a fair say and the
neighbour matches on age, visits and channel too. Largest raw spreads:
{', '.join(f'{k} {v:,.0f}' for k, v in spread.sort_values(ascending=False).head(3).items())}.""")

step("2.2  Size vs mix: Euclidean vs cosine")
spend = customers[SPEND_COLS].to_numpy(float)
total = spend.sum(axis=1)
big, small = int(np.argmax(total)), None
mix = spend / total[:, None]
small = int(np.argmin(np.where(total < np.quantile(total, 0.2), np.abs(mix - mix[big]).sum(1), np.inf)))
pair = customers.loc[[big, small], SPEND_COLS].assign(total=total[[big, small]])
print(pair.round(0).to_string())
show("Euclidean distance between them", f"{euclidean(spend[big], spend[small]):,.0f}")
show("Cosine similarity between them", f"{cosine_sim(spend[big], spend[small]):.3f}")
rank_e = (cdist(spend[[big]], spend)[0] < euclidean(spend[big], spend[small])).sum()
show("How many customers are 'closer' by Euclidean distance", f"{rank_e:,} of {len(spend):,}")
note("""A big spender and a small spender who buy the SAME MIX of categories: cosine
says nearly identical, Euclidean says far apart (most customers are closer).
For 'people with similar tastes' (recommendations, segment by mix) use cosine;
for 'people who spend similar amounts' (credit limits, value tiers) use a
distance. This is also why cosine is the default for TEXT: a long and a short
document about the same topic point the same way.""")

step("2.3  Text: document length")
docs = ["refund policy for damaged items",
        "refund policy for damaged items. " * 5 + "our refund policy covers damaged items and late delivery",
        "how to reset my password"]
V = CountVectorizer().fit_transform(docs).toarray()
show("Euclidean: short vs long refund doc / short refund vs password doc",
     f"{euclidean(V[0], V[1]):.2f} / {euclidean(V[0], V[2]):.2f}")
show("Cosine:    short vs long refund doc / short refund vs password doc", f"{cosine_sim(V[0], V[1]):.2f} / {cosine_sim(V[0], V[2]):.2f}")
note("""By Euclidean distance the short refund note is CLOSER to the unrelated password
note than to the long refund page, simply because the long page has bigger
word counts. Cosine fixes this: same topic, same direction. Search engines
and embedding retrieval (Labs 08-10) compare documents by cosine.""")

# %% =========================================================================
# LEVEL 3 (HARD): High dimensions, correlation, learned vectors
# ============================================================================
banner("When distances stop meaning much, and when they start meaning more", "HARD")
step("3.1  The curse of dimensionality")
rng = np.random.default_rng(0)
rows = {}
for d in [2, 10, 100, 1000, 5000]:
    P = rng.uniform(size=(500, d))
    dist = cdist(P[:50], P)
    dist[dist == 0] = np.nan
    rows[d] = {"nearest / farthest distance": float(np.nanmean(np.nanmin(dist, 1) / np.nanmax(dist, 1))),
               "relative spread of distances (sd / mean)": float(np.nanmean(np.nanstd(dist, 1) / np.nanmean(dist, 1)))}
curse = pd.DataFrame(rows).T
curse.index.name = "dimensions"
print(curse.round(3).to_string())
note(f"""With random points, as dimensions grow every point becomes almost equally far
from every other: the nearest neighbour is {curse.iloc[-1, 0]:.0%} as far away as the farthest
in 5,000 dimensions. 'Nearest' stops meaning 'similar'. This is why raw
high-dimensional data (bag-of-words, pixels) is first compressed (PCA, Lab 05)
or turned into LEARNED dense vectors (embeddings, Lab 08), where distances
reflect meaning instead of noise.""")

step("3.2  Correlated features count twice")
dup = np.c_[X_std, X_std[:, [2]], X_std[:, [2]]]                   # groceries arrives from 3 data sources
probe = np.arange(300)


def nearest(Z, metric="euclidean", **kw):
    d = cdist(Z[probe], Z, metric=metric, **kw)
    d[np.arange(len(probe)), probe] = np.inf                          # a customer is not their own neighbour
    return d.argmin(axis=1)


changed_euc = np.mean(nearest(X_std) != nearest(dup))
changed_mah = np.mean(nearest(X_std, "mahalanobis", VI=np.linalg.inv(np.cov(X_std.T)))
                      != nearest(dup, "mahalanobis", VI=np.linalg.pinv(np.cov(dup.T))))
show("Euclidean: nearest neighbours that change when groceries is copied 3x", f"{changed_euc:.0%} of 300 customers")
show("Mahalanobis: nearest neighbours that change", f"{changed_mah:.0%}")
note(f"""If three columns all measure grocery spend (three data sources), Euclidean
distance counts that one aspect three times: {changed_euc:.0%} of customers get a different
nearest neighbour, now chosen mostly for a similar grocery bill. Mahalanobis
distance divides out the correlations (it uses the inverse covariance), so
duplicated information counts once: its neighbours don't change ({changed_mah:.0%}).
Remove redundant features, or use a correlation-aware distance.""")

step("3.3  Similarity of LEARNED vectors (a preview of embeddings)")
emb = PPMIEmbeddings(dim=50).fit(make_text_corpus("hr"))
pairs = [("vacation", "holiday"), ("salary", "wages"), ("supervisor", "manager"), ("vacation", "salary"), ("doctor", "pension")]
for w1, w2 in pairs:
    v1, v2 = emb.vectors_[emb.index_[w1]], emb.vectors_[emb.index_[w2]]
    show(f"cosine('{w1}', '{w2}')", f"{cosine_sim(v1, v2):+.3f}")
note("""Words are not numbers, but we can LEARN a vector for each word from how it is
used (Lab 08). Then cosine similarity of those vectors measures similarity of
MEANING: 'vacation' and 'holiday' point the same way, 'vacation' and 'salary'
do not. Every modern semantic search, recommender and RAG system rests on this
one operation: cosine similarity between learned vectors.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(curse.index, curse.iloc[:, 0], "o-")
    axes[0].set(xscale="log", xlabel="dimensions (log)", ylabel="nearest / farthest", title="The curse of dimensionality")
    axes[1].scatter(total, mix[:, 0], s=4, alpha=0.4)
    axes[1].scatter(total[[big, small]], mix[[big, small], 0], s=80, color="r")
    axes[1].set(xscale="log", xlabel="total monthly spend (log)", ylabel="share spent on groceries",
                title="Same mix, very different size (red)")
    w = ["vacation", "holiday", "leave", "salary", "wages", "pay", "manager", "supervisor", "doctor", "pension"]
    S = np.array([[float(emb.vectors_[emb.index_[x]] @ emb.vectors_[emb.index_[yy]]) for yy in w] for x in w])
    axes[2].imshow(S, cmap="Blues", vmin=-0.2, vmax=1)
    axes[2].set_xticks(range(len(w)), w, rotation=60, fontsize=8)
    axes[2].set_yticks(range(len(w)), w, fontsize=8)
    axes[2].set_title("Cosine similarity of learned word vectors")
    save_fig(fig, "02_distance_similarity")

banner("KEY TAKEAWAYS")
print("""  1. Euclidean / Manhattan measure HOW MUCH things differ; cosine measures the angle (the mix).
  2. Jaccard compares sets (baskets, tags); Hamming compares yes/no profiles.
  3. Scale features (and log skewed ones) or the biggest numbers decide who is 'similar'.
  4. In high dimensions raw distances concentrate; correlated features count twice (Mahalanobis fixes it).
  5. Cosine similarity between LEARNED vectors measures similarity of meaning: the core of semantic search.""")
