"""
LAB 09 - HANDS-ON LAB: A Semantic Product Search and 'More Like This' System
============================================================================

THE PROBLEM
    An online shop's search box only matches exact words: customers who type 'earbuds' never see
    the products described as 'headphones', 'headset' or 'earphones'. The team wants
    (a) semantic search: find the right products whatever words the customer uses, and
    (b) 'more like this': similar products on every product page.

THE DATA: a catalogue of 600 products in 10 kinds, written by different suppliers with different
    SYNONYMS ('sofa', 'couch', 'settee'...), plus the shop's historical catalogue text (unlabelled)
    to learn embeddings from. 26 real-looking shopper queries with the product kind they want.

WHAT YOU WILL DO (the syllabus hands-on lab)
    Part 1 (EASY)   PREPARE the product data
    Part 2 (EASY)   GENERATE FEATURE VECTORS: TF-IDF, character n-grams, learned embeddings
    Part 3 (MEDIUM) CALCULATE COSINE SIMILARITY; RETRIEVE SIMILAR ITEMS ('more like this'), evaluated
    Part 4 (MEDIUM) Semantic SEARCH for shopper queries; a tuned hybrid
    Part 5 (HARD)   VISUALISE CLUSTERS USING PCA; near-duplicates; filters; failure analysis; save the index
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

from ml_lab_utils import HAS_MPL, OUT_DIR, banner, make_product_catalogue, note, plt, save_fig, show, step
from unsup_toolkit import FlatIndex, PPMIEmbeddings, tokenize

# %% =========================================================================
# PART 1 (EASY): Prepare the data
# ============================================================================
banner("PART 1 - Prepare the product data", "EASY")
catalogue = make_product_catalogue(per_sub=60, seed=9)
show("Products / kinds (subcategories) / categories", f"{len(catalogue)} / {catalogue.subcategory.nunique()} / {catalogue.category.nunique()}")
print(catalogue.head(4)[["product_id", "title", "description", "subcategory", "price"]].to_string())
catalogue["text"] = (catalogue.title + ". " + catalogue.title + ". " + catalogue.description).str.lower()   # title counted twice
catalogue["text"] = catalogue.text.str.replace(r"(free delivery|two year warranty|best seller|new arrival)\.?", "", regex=True)
dupes = catalogue.duplicated(subset=["text"]).sum()
show("Exact duplicate texts", int(dupes))
nouns = {sub: set() for sub in catalogue.subcategory.unique()}
for sub, txt in zip(catalogue.subcategory, catalogue.description):
    nouns[sub].update(tokenize(txt.split(" offers ")[0].replace("This ", "")))
print("  Words suppliers use for the SAME kind of product:")
for sub in ["headphones", "sofa", "running shoes"]:
    print(f"    {sub:<14}: {sorted(nouns[sub])}")
note("""Preparation decisions: lower-case everything; count the title twice (it's what
the supplier thought mattered most); strip marketing boilerplate ('free
delivery', 'best seller') that says nothing about WHAT the product is. And see
the core difficulty: one kind of product is described with many different
words, so exact-word search fails silently.""")

# %% =========================================================================
# PART 2 (EASY): Feature vectors
# ============================================================================
banner("PART 2 - Generate feature vectors", "EASY")
history = make_product_catalogue(per_sub=300, seed=99, nouns_per_product=2)   # past listings + reviews: unlabelled text
corpus = (history.title + ". " + history.description).str.lower().tolist()
word_tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(corpus + catalogue.text.tolist())
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(corpus + catalogue.text.tolist())
emb = PPMIEmbeddings(dim=60, window=5).fit(corpus)
idf = dict(zip(word_tfidf.get_feature_names_out(), word_tfidf.idf_))
REPS = {
    "word TF-IDF": lambda texts: word_tfidf.transform(texts).toarray(),
    "character 3-5-gram TF-IDF": lambda texts: char_tfidf.transform(texts).toarray(),
    "learned embeddings (PPMI+SVD, 60 dims)": lambda texts: emb.embed(texts, idf),
}
V = {name: f(catalogue.text.tolist()) for name, f in REPS.items()}
for name, M in V.items():
    show(f"{name}: dimensions / non-zero share", f"{M.shape[1]:,} / {np.mean(M != 0):.1%}")
show("Embedding neighbours of 'earbuds'", ", ".join(w for w, _ in emb.most_similar("earbuds", 5)))
show("Embedding neighbours of 'couch'", ", ".join(w for w, _ in emb.most_similar("couch", 5)))
note("""Three ways to turn a product into a vector. Word TF-IDF: thousands of dimensions,
almost all zero, exact words only. Character n-grams: robust to typos and word
forms ('earbud' ~ 'earbuds'), still spelling-based. Learned embeddings: 60
dense numbers per product, learned from 3,000 past listings without labels;
'earbuds' lands next to 'headphones', 'couch' next to 'sofa', because suppliers
use them in the same contexts.""")

# %% =========================================================================
# PART 3 (MEDIUM): Cosine similarity and 'more like this'
# ============================================================================
banner("PART 3 - Cosine similarity and 'more like this'", "MEDIUM")
sub = catalogue.subcategory.to_numpy()


def neighbours(M, i, k=5):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sims = Mn @ Mn[i]
    sims[i] = -np.inf
    top = np.argsort(-sims)[:k]
    return top, sims[top]


item = int(np.flatnonzero(sub == "headphones")[3])
print(f"  Product: {catalogue.title[item]}  [{sub[item]}]")
for name, M in V.items():
    top, sims = neighbours(M, item)
    print(f"    {name}:")
    for t, s_ in zip(top, sims):
        print(f"       {s_:.2f}  {catalogue.title[t]:<45} [{sub[t]}]{'' if sub[t] == sub[item] else '  <- wrong kind'}")


PER_KIND = int(pd.Series(sub).value_counts().iloc[0])                 # 60 products of every kind


def mlt_scores(M):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    S_ = Mn @ Mn.T
    np.fill_diagonal(S_, -np.inf)
    order = np.argsort(-S_, axis=1)
    same = sub[order] == sub[:, None]
    return {"precision@10 (10 most similar)": same[:, :10].mean(),
            f"R-precision (top {PER_KIND - 1}: how much of the kind is reachable)": same[:, :PER_KIND - 1].mean()}


mlt = pd.DataFrame({name: mlt_scores(M) for name, M in V.items()}).T
print(mlt.round(3).to_string())
note(f"""'More like this' = the products with the highest cosine similarity to this one.
All three look perfect on the first 10 neighbours: with 60 products per kind,
there are always 10 that share the exact words. The real difference shows when
we ask how much of the kind is REACHABLE (R-precision: of the {PER_KIND - 1} most similar
products, how many are the same kind?). Word TF-IDF: {mlt.iloc[0, 1]:.0%}; a headphone page
never shows the products called 'cans' or 'headset'. Learned embeddings: {mlt.iloc[2, 1]:.0%}.""")

# %% =========================================================================
# PART 4 (MEDIUM): Semantic search
# ============================================================================
banner("PART 4 - Semantic search for shopper queries", "MEDIUM")
QUERIES = [("wireless earbuds for the gym", "headphones"), ("noise cancelling cans", "headphones"), ("bluetooth headset for calls", "headphones"),
           ("lightweight ultrabook for travel", "laptop"), ("portable computer with long battery", "laptop"), ("notebook with fast processor", "laptop"),
           ("fitness tracker with heart rate", "smartwatch"), ("waterproof wearable for swimming", "smartwatch"), ("gps activity watch", "smartwatch"),
           ("marathon trainers", "running shoes"), ("trail sneakers with good grip", "running shoes"), ("breathable joggers", "running shoes"),
           ("non-slip pilates mat", "yoga mat"), ("thick exercise mat", "yoga mat"),
           ("two person backpacking shelter", "tent"), ("waterproof camping tent", "tent"),
           ("espresso machine with milk frother", "coffee maker"), ("single serve brewer", "coffee maker"), ("barista style coffee machine", "coffee maker"),
           ("smoothie maker for crushing ice", "blender"), ("portable mixer", "blender"),
           ("velvet couch", "sofa"), ("corner settee", "sofa"), ("loveseat sofa bed", "sofa"),
           ("dimmable reading light", "desk lamp"), ("led task light with usb charging", "desk lamp")]
qtext = [q for q, _ in QUERIES]
qsub = np.array([s for _, s in QUERIES])


def search_scores(name):
    Q, M = REPS[name](qtext), V[name]
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Qn @ Mn.T


def p_at_10(S):
    top = np.argsort(-S, axis=1)[:, :10]
    return np.mean(sub[top] == qsub[:, None], axis=1)


def r_prec(S):
    top = np.argsort(-S, axis=1)[:, :PER_KIND]
    return np.mean(sub[top] == qsub[:, None], axis=1)


S = {name: search_scores(name) for name in REPS}
val, tst = np.arange(0, len(QUERIES), 2), np.arange(1, len(QUERIES), 2)     # tune the blend on half the queries
best_w, best_v = 0.0, -1
for w in np.linspace(0, 1, 11):
    hyb = w * S["word TF-IDF"] + (1 - w) * S["learned embeddings (PPMI+SVD, 60 dims)"]
    v_ = r_prec(hyb)[val].mean()
    if v_ > best_v:
        best_w, best_v = w, v_
S[f"hybrid ({best_w:.1f} x TF-IDF + {1 - best_w:.1f} x embeddings, tuned)"] = (
    best_w * S["word TF-IDF"] + (1 - best_w) * S["learned embeddings (PPMI+SVD, 60 dims)"])
search_tab = pd.DataFrame({name: {"precision@10": p_at_10(s_).mean(), "R-precision (all queries)": r_prec(s_).mean(),
                                  "R-precision (held-out half)": r_prec(s_)[tst].mean()} for name, s_ in S.items()}).T
print(search_tab.round(3).to_string())
q_i = qtext.index("velvet couch")
for name in ["word TF-IDF", "learned embeddings (PPMI+SVD, 60 dims)"]:
    top = np.argsort(-S[name][q_i])[:3]
    print(f"  'velvet couch', {name}: " + " | ".join(f"{catalogue.title[t]} [{sub[t]}]" for t in top))
note(f"""26 queries in shopper language. Again the top 10 looks fine for every method
(there are always 10 products with the query's exact words), but a shopper who
types 'couch' only ever sees the sofas whose supplier ALSO wrote 'couch'. Word
TF-IDF surfaces {search_tab.iloc[0, 1]:.0%} of the relevant stock in the first {PER_KIND} results; the learned
embeddings {search_tab.iloc[2, 1]:.0%}. For a shop, the unreachable third of the stock is lost sales.
The hybrid weight was tuned on half the queries and checked on the other half
(as with any hyperparameter: Module 5, Lab 07); here pure embeddings win.""")

# %% =========================================================================
# PART 5 (HARD): Clusters, duplicates, filters, failures, the index
# ============================================================================
banner("PART 5 - Visualise clusters, find duplicates, add filters, save the index", "HARD")
step("5.1  Visualise clusters with PCA")
E = V["learned embeddings (PPMI+SVD, 60 dims)"]
pca = PCA(2).fit(E)
P2 = pca.transform(E)
km = KMeans(10, n_init=10, random_state=0).fit(E)
show("Variance kept by 2 PCA components", f"{pca.explained_variance_ratio_.sum():.0%}")
show("k-means (k=10) on embeddings vs true kinds (ARI)", f"{adjusted_rand_score(sub, km.labels_):.3f}")
show("k-means (k=10) on word TF-IDF vs true kinds (ARI)",
     f"{adjusted_rand_score(sub, KMeans(10, n_init=10, random_state=0).fit_predict(V['word TF-IDF'])):.3f}")
note("""Projected to 2-D with PCA (outputs/09_product_map.png), products of the same kind
form visible islands, although 2 components keep only a quarter of the
variance. k-means on the vectors rediscovers the 10 product kinds without any
labels: a catalogue taxonomy for free, useful for supplier products that arrive
uncategorised. (Here word TF-IDF clusters perfectly too: each kind also has its
own feature phrases. Clustering is easier than search, because a cluster only
needs most of its members to share SOME words, not the query's words.)""")

step("5.2  Near-duplicate listings")
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
W = V["word TF-IDF"] / (np.linalg.norm(V["word TF-IDF"], axis=1, keepdims=True) + 1e-12)
sim_w = W @ W.T
np.fill_diagonal(sim_w, 0)
pairs = np.argwhere(np.triu(sim_w > 0.9))
show("Pairs with word-TF-IDF cosine > 0.9 (near-identical text)", len(pairs))
for a, b in pairs[:3]:
    print(f"    {catalogue.title[a]}  ~  {catalogue.title[b]}   (prices {catalogue.price[a]} / {catalogue.price[b]})")
note("""Very high cosine similarity on the WORDS flags near-duplicate listings (same
supplier template, maybe the same product listed twice): candidates to merge.
For duplicates use exact-word similarity with a high threshold; for 'similar
products' use embeddings. Different questions, different vectors.""")

step("5.3  Search with filters: 'running shoes under $50'")
index = FlatIndex().add(E, catalogue.product_id)
q_vec = emb.embed(["comfortable trainers for running"], idf)
t0 = time.perf_counter()
hits = index.search(q_vec, k=len(catalogue))
latency = (time.perf_counter() - t0) * 1000
lookup = catalogue.set_index("product_id")
filtered = [(pid, s_) for pid, s_ in hits if lookup.price[pid] < 50][:5]
for pid, s_ in filtered:
    print(f"    {s_:.2f}  {lookup.title[pid]:<45} ${lookup.price[pid]:.2f}  [{lookup.subcategory[pid]}]")
show("Search latency over 600 products (flat index)", f"{latency:.2f} ms")
note("""Real search combines similarity with structured FILTERS (price, brand, stock).
Here we rank by similarity, then keep only products under $50: a 'post-filter'.
Vector databases also support PRE-filtering (search only inside the filter),
which matters when the filter is very selective. Lab 10 builds the index
structure that keeps this fast at millions of products.""")

step("5.4  Failure analysis: which queries still fail?")
emb_p = r_prec(S["word TF-IDF"])
fails = pd.DataFrame({"query": qtext, "wanted": qsub, "TF-IDF R-precision": emb_p,
                      "embedding R-precision": r_prec(S["learned embeddings (PPMI+SVD, 60 dims)"])}).sort_values("TF-IDF R-precision").head(4)
fails["TF-IDF's most common wrong kind"] = [
    pd.Series([k_ for k_ in sub[np.argsort(-S["word TF-IDF"][qtext.index(q)])[:PER_KIND]] if k_ != w]).mode().iloc[0]
    if any(k_ != w for k_ in sub[np.argsort(-S["word TF-IDF"][qtext.index(q)])[:PER_KIND]]) else "-"
    for q, w in zip(fails["query"], fails["wanted"])]
print(fails.round(2).to_string(index=False))
note("""Read the failures, not just the average. TF-IDF's worst queries are the ones
whose noun most suppliers don't use, plus generic words ('portable', 'compact',
'long battery') that many kinds share, so the remaining slots fill up with
the wrong kind. Embeddings fix most of these; what remains is fixed with
data-centric work: query understanding (detect the product noun), synonyms
mined from search logs, or re-ranking with category signals (Module 5, Lab 09).""")

step("5.5  Save the index")
np.savez_compressed(OUT_DIR / "product_index.npz", vectors=E, ids=catalogue.product_id.to_numpy())
catalogue[["product_id", "title", "subcategory", "price"]].to_csv(OUT_DIR / "product_index_metadata.csv", index=False)
print("  [saved] outputs/product_index.npz (60-dim vectors + ids) and product_index_metadata.csv")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for k_ in np.unique(sub):
        m_ = sub == k_
        axes[0].scatter(P2[m_, 0], P2[m_, 1], s=10, label=k_)
    axes[0].legend(fontsize=7, markerscale=2)
    axes[0].set(title="Product embeddings, PCA to 2-D (colour = true kind)", xlabel="PC1", ylabel="PC2")
    st = search_tab.iloc[:, 1]
    axes[1].barh(st.index, st.values, color=["#9d9d9d", "#4c78a8", "#54a24b", "#f58518"])
    axes[1].set(xlim=(0, 1), xlabel="R-precision on 26 shopper queries (share of the kind reachable)", title="Search quality")
    axes[1].tick_params(axis="y", labelsize=8)
    save_fig(fig, "09_product_map")

banner("KEY TAKEAWAYS")
print("""  1. Prepare text deliberately: lower-case, weight the title, strip boilerplate, check duplicates.
  2. Vectors: TF-IDF (exact words), character n-grams (spelling), learned embeddings (meaning).
  3. Cosine similarity powers both 'more like this' and search; evaluate both against known kinds.
  4. Tune hybrid weights on held-out queries; add structured filters; read the failing queries.
  5. PCA maps and k-means on embeddings reveal the catalogue's structure without labels.""")
