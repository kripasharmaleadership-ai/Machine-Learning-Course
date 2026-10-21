"""
LAB 06 - Embedding Generation and Vector Search
===============================================

FROM TEXT TO VECTORS (Module 6, now in service of retrieval)
    An EMBEDDING maps a text to a vector so that texts with similar MEANING land close together. Then
    "find the article that answers this question" becomes "find the nearest vectors": vector search.

    keyword (TF-IDF / BM25)   one dimension per word; two texts are similar if they share (rare) words
    semantic (embeddings)     dense vectors learned from co-occurrence; 'refund' sits near 'money back'
    hybrid                    both: cosine = alpha * semantic + (1 - alpha) * keyword

    Our semantic model is small and home-made (PPMI + SVD on 6,000 support sentences, as in Module 6), so
    every result here is visible and reproducible. Production systems use a pretrained embedding model
    (an API, or an open model) with hundreds of dimensions: better vectors, the SAME pipeline and metrics.

THE TEST
    144 test tickets whose answer lives in a known help-centre article (kb_ids), plus two small hand-written
    sets: 10 PARAPHRASES (no shared words with the article) and 10 EXACT-TERM questions ('C-wire', '9V').
    recall@k = share of questions with a right article in the top k.  MRR = average of 1 / rank of the first hit.

LEVELS
    EASY   : What the embedding learned: word neighbours and sentence similarity, keyword vs semantic.
    MEDIUM : Keyword vs semantic vs hybrid retrieval, measured. Tuning alpha on dev, checking on test.
    HARD   : Vector search at scale (exact vs approximate), and the plot twist: the best upgrade isn't a
             better embedding.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from genai_toolkit import HybridEmbedder, PPMIEmbeddings, TfidfEmbedder, VectorIndex, kb_chunks, retrieval_report
from ml_lab_utils import HAS_MPL, banner, make_kb, make_support_corpus, make_tickets, note, plt, save_fig, show, step

kb = make_kb()
items = kb_chunks(kb, max_words=None)                                # one chunk per article, with title (Lab 05)
tickets = make_tickets()
kbq = tickets[tickets.kb_ids.map(len) > 0]
dev, test = kbq[kbq.split == "dev"], kbq[kbq.split == "test"]
corpus = make_support_corpus()
semantic = PPMIEmbeddings(dim=60).fit(corpus)
keyword = TfidfEmbedder().fit([i["text"] for i in items] + make_support_corpus(n=500))
P = "LM-KB-"
PARAPHRASES = [("When will the money show up after you get my parcel back?", "REF"),
               ("Is there a guarantee if my purchase turns out faulty?", "WAR"),
               ("I need to call off my purchase before it ships", "CAN"),
               ("How long until my reimbursement arrives?", "REF"),
               ("Can I send it back if it's faulty?", "RET"),
               ("I forgot my login, what now?", "PWD"),
               ("My purchase got cheaper a week later", "PRM"),
               ("What does the guarantee cover?", "WAR"),
               ("How many days until my shipment gets here?", "SHP"),
               ("Can you stop my order, I bought the wrong one", "CAN")]
EXACT = [("thermostat C-wire", "T04"), ("2.4 GHz Wi-Fi hub", "T01"), ("9V battery keypad", "T05"),
         ("infrared night vision", "T03"), ("Lumora Protect price", "SUB"), ("restocking fee opened items", "RET"),
         ("transformer 16 volts doorbell", "T06"), ("backup codes two-step verification", "PWD"),
         ("flash sales price match", "PRM"), ("brush roll dust bin", "T07")]


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def evaluate(embedder, queries, gold, index_items=items, k=10):
    index = VectorIndex(embedder).add(index_items)
    ranked = []
    for q in queries:
        ids = []
        for h in index.search(q, k=k):
            if h["doc_id"] not in ids:
                ids.append(h["doc_id"])
        ranked.append(ids)
    return retrieval_report(ranked, gold)


# %% =========================================================================
# LEVEL 1 (EASY): What the embedding learned
# ============================================================================
banner("What the embedding learned", "EASY")
show("Vocabulary / dimensions", f"{len(semantic.vocab_):,} words x {semantic.vectors_.shape[1]}")
rows = {}
for w in ["refund", "broken", "delivery", "cancel", "password", "warranty"]:
    sims = semantic.vectors_ @ semantic.vectors_[semantic.index_[w]]
    rows[w] = [semantic.vocab_[i] for i in np.argsort(-sims)[1:6]]
print(pd.DataFrame(rows, index=[f"#{i}" for i in range(1, 6)]).T.to_string())
note("""Nobody told the model that 'refund' and 'reimbursement' mean the same thing,
or that 'broken' ~ 'faulty' ~ 'dead'. It learned it from which words appear
in the same contexts (Module 6). Not every neighbour is a synonym ('want',
'need'): co-occurrence also links words that merely appear together.""")

step("1.1  Sentence similarity: keyword vs semantic")
pairs = [("I want my money back", "Can I get a refund?"),
         ("My parcel never arrived", "Where is my delivery?"),
         ("The bulb is faulty", "The bulb is broken"),
         ("Cancel my order", "Change my order's address"),
         ("How do I reset my password?", "What's the weather this weekend?")]
rows = []
for a, b in pairs:
    Ka, Kb = keyword.embed([a, b])
    Sa, Sb = semantic.embed([a, b])
    rows.append({"text A": a, "text B": b, "keyword cosine": cosine(Ka, Kb), "semantic cosine": cosine(Sa, Sb)})
sim_table = pd.DataFrame(rows)
print(sim_table.round(2).to_string(index=False))
note(f"""'I want my money back' and 'Can I get a refund?' share no words: keyword
similarity {sim_table.iloc[0, 2]:.2f}, semantic {sim_table.iloc[0, 3]:.2f}. That is the promise of embeddings. But look at
'Cancel my order' vs 'Change my order's address': {sim_table.iloc[3, 3]:.2f}, {'even HIGHER than' if sim_table.iloc[3, 3] > sim_table.iloc[0, 3] else 'close to'} the
refund pair, because both are about orders. Embeddings measure 'about the
same topic', which is not the same as 'answers the same question'.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Keyword vs semantic vs hybrid, measured
# ============================================================================
banner("Keyword vs semantic vs hybrid retrieval", "MEDIUM")
retrievers = {"keyword (TF-IDF)": keyword, "semantic (PPMI)": semantic, "hybrid (alpha 0.5)": HybridEmbedder(keyword, semantic, 0.5)}
rows = {}
for name, emb in retrievers.items():
    rows[(name, "144 test tickets")] = evaluate(emb, list(test.text), list(test.kb_ids))
    rows[(name, "10 paraphrases")] = evaluate(emb, [q for q, _ in PARAPHRASES], [[P + g] for _, g in PARAPHRASES])
    rows[(name, "10 exact terms")] = evaluate(emb, [q for q, _ in EXACT], [[P + g] for _, g in EXACT])
board = pd.DataFrame(rows).T
board.index.names = ["retriever", "questions"]
print(board.round(3).to_string())
mrr = board["MRR"].unstack()
note(f"""On the real tickets, plain KEYWORD search wins (MRR {mrr.loc['keyword (TF-IDF)', '144 test tickets']:.2f} vs semantic {mrr.loc['semantic (PPMI)', '144 test tickets']:.2f}).
Support tickets reuse the help centre's own words ('return', 'warranty',
'password'), and exact matches on rare words are hard to beat. Semantic
search earns a small edge on paraphrases ({mrr.loc['semantic (PPMI)', '10 paraphrases']:.2f} vs {mrr.loc['keyword (TF-IDF)', '10 paraphrases']:.2f}) and loses badly on
exact terms like 'C-wire' and '9V' ({mrr.loc['semantic (PPMI)', '10 exact terms']:.2f} vs {mrr.loc['keyword (TF-IDF)', '10 exact terms']:.2f}): an averaged vector blurs the one
rare word that mattered. Hybrid keeps keyword's precision on exact terms. A
much better embedding model would narrow the gap; it wouldn't make the
keyword baseline obsolete. Always measure the boring baseline first.""")

step("2.1  Tuning alpha on dev, checking on test")
rows = {}
for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    emb = HybridEmbedder(keyword, semantic, a)
    rows[a] = {"dev MRR": evaluate(emb, list(dev.text), list(dev.kb_ids))["MRR"],
               "test MRR": evaluate(emb, list(test.text), list(test.kb_ids))["MRR"]}
alpha = pd.DataFrame(rows).T
alpha.index.name = "alpha (0 = keyword only, 1 = semantic only)"
print(alpha.round(3).to_string())
best_a = alpha["dev MRR"].idxmax()
note(f"""Choose alpha on DEV ({best_a}), then look at TEST once: {alpha.loc[best_a, 'test MRR']:.3f}. Here the tuning
says what the table above said: on these tickets, more weight on keywords is
better. With a strong pretrained embedding model the optimum usually moves
towards the middle; the point is to measure it on your own questions.""")

# %% =========================================================================
# LEVEL 3 (HARD): Vector search at scale, and the plot twist
# ============================================================================
banner("Vector search at scale: exact vs approximate", "HARD")
rng = np.random.default_rng(0)
d, n = 64, 200_000
centers = rng.normal(size=(300, d))
X = centers[rng.integers(300, size=n)] + 1.4 * rng.normal(size=(n, d))     # 200k 'chunk embeddings' in topic clusters
X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = centers[rng.integers(300, size=200)] + 1.4 * rng.normal(size=(200, d))
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
t0 = time.perf_counter()
exact = [np.argpartition(-(X @ q), 10)[:10] for q in Q]
brute_ms = (time.perf_counter() - t0) / len(Q) * 1000
t0 = time.perf_counter()
km = KMeans(256, n_init=1, random_state=0, max_iter=20).fit(X[:50_000])
cells = km.predict(X)
lists = [np.where(cells == c)[0] for c in range(256)]
build_s = time.perf_counter() - t0
rows = {"exact (brute force)": {"recall@10 vs exact": 1.0, "ms per query": brute_ms, "vectors scanned": n}}
for nprobe in [1, 2, 4, 8, 16, 32]:
    t0, rec, scanned = time.perf_counter(), [], []
    for q, truth in zip(Q, exact):
        cand = np.concatenate([lists[c] for c in np.argsort(-(km.cluster_centers_ @ q))[:nprobe]])
        top = cand[np.argpartition(-(X[cand] @ q), min(10, len(cand) - 1))[:10]]
        rec.append(len(set(top) & set(truth)) / 10)
        scanned.append(len(cand))
    rows[f"IVF, search {nprobe} of 256 cells"] = {"recall@10 vs exact": np.mean(rec),
                                                 "ms per query": (time.perf_counter() - t0) / len(Q) * 1000,
                                                 "vectors scanned": np.mean(scanned)}
ann = pd.DataFrame(rows).T
print(ann.round(3).to_string())
show("IVF index build time (k-means on 50k vectors, assign 200k)", f"{build_s:.1f} s")
note(f"""Exact search compares the question with every vector: {n:,} dot products in
{brute_ms:.1f} ms with NumPy. An inverted-file (IVF) index clusters the vectors once, then
searches only the cells nearest the question: scanning ~{ann.iloc[1, 2]:,.0f} vectors is {brute_ms / ann.iloc[1, 1]:.0f}x faster
but finds only {ann.iloc[1, 0]:.0%} of the true top 10; more cells, better recall, less speed.
That's the knob behind every vector database (FAISS, HNSW-based stores, ...).
For a help centre of a few thousand chunks, exact search is instant and
exactly right; approximate search pays off at millions of vectors.""")

step("3.1  Plot twist: index the customers' words, not just the articles'")
extra = []
for r in dev.itertuples():                                           # solved DEV tickets, attached to their article
    for doc_id in r.kb_ids:
        art = next(i for i in items if i["doc_id"] == doc_id)
        extra.append({**art, "chunk_id": f"{art['chunk_id']}#q{r.Index}", "text": r.text})
keyword_plus = TfidfEmbedder().fit([i["text"] for i in items + extra] + make_support_corpus(n=500))
rows = {"hybrid, articles only": evaluate(HybridEmbedder(keyword, semantic, 0.5), list(test.text), list(test.kb_ids)),
        "keyword, articles + past questions": evaluate(keyword_plus, list(test.text), list(test.kb_ids), items + extra),
        "hybrid, articles + past questions": evaluate(HybridEmbedder(keyword_plus, semantic, 0.5), list(test.text),
                                                      list(test.kb_ids), items + extra)}
plus = pd.DataFrame(rows).T
print(plus.round(3).to_string())
idx_a = VectorIndex(HybridEmbedder(keyword, semantic, 0.5)).add(items)
idx_b = VectorIndex(HybridEmbedder(keyword_plus, semantic, 0.5)).add(items + extra)
q = "How fast is delivery normally?"
show(f"'{q}' -> top article, before", idx_a.search(q, k=1)[0]["title"])
show(f"'{q}' -> top article, after", idx_b.search(q, k=1)[0]["title"])
note(f"""We didn't change the embedding at all. We indexed {len(extra)} solved DEV tickets next to
the article that answered them, so the index now also contains the words
CUSTOMERS use ('how fast is delivery') pointing at the article that uses the
company's words ('standard shipping takes'). Test MRR: {plus.iloc[0, -1]:.2f} -> {plus.iloc[2, -1]:.2f}. Caveat: our
tickets come from templates, so test questions resemble dev questions more
than real ones would, and the real gain is smaller. But the lesson holds and
is used in production (it's called document expansion, or 'doc2query'):
the cheapest retrieval upgrade is often your own history, not a new model.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    mrr.plot.bar(ax=axes[0], rot=0)
    axes[0].set(ylabel="MRR", ylim=(0, 1.05), title="Keyword vs semantic vs hybrid")
    axes[0].legend(fontsize=7)
    axes[1].plot(alpha.index, alpha["dev MRR"], "o-", label="dev")
    axes[1].plot(alpha.index, alpha["test MRR"], "s--", label="test")
    axes[1].set(xlabel="alpha (0 = keyword, 1 = semantic)", ylabel="MRR", title="Tuning the hybrid weight")
    axes[1].legend()
    a = ann.iloc[1:]
    axes[2].plot(a["ms per query"], a["recall@10 vs exact"], "o-", color="C2", label="IVF (1 to 32 cells)")
    axes[2].scatter([brute_ms], [1.0], color="C3", s=80, zorder=3, label="exact")
    axes[2].set(xlabel="ms per query", ylabel="recall@10 vs exact", title="Approximate search: speed vs recall (200k vectors)")
    axes[2].legend()
    save_fig(fig, "06_embeddings_and_vector_search")

banner("KEY TAKEAWAYS")
print("""  1. Embeddings put similar meanings close together; vector search = nearest neighbours.
  2. 'About the same topic' is not 'answers the same question': measure retrieval with recall@k and MRR.
  3. Keyword search is a strong baseline, especially for exact terms; hybrid keeps the best of both.
  4. Tune weights on dev, report on test.
  5. Exact search is fine for thousands of chunks; approximate (IVF/HNSW) trades recall for speed at scale.
  6. Indexing your customers' past questions can beat a better embedding model.""")
