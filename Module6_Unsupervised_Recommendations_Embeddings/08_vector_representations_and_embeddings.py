"""
LAB 08 - Vector Representations and Embeddings
==============================================

EVERY MODEL NEEDS NUMBERS. HOW DO WE TURN TEXT (OR ITEMS, OR USERS) INTO VECTORS?
    one-hot          one dimension per word in the vocabulary; a word = a single 1      (sparse, no meaning)
    bag-of-words     a document = word counts                                            (sparse, ~vocabulary size)
    TF-IDF           counts re-weighted: rare, specific words count more                 (sparse; the classic search engine)
    EMBEDDINGS       a short DENSE vector (here 50 numbers) LEARNED so that things used
                     in similar contexts get similar vectors                             (dense; meaning, not spelling)

WHERE EMBEDDINGS COME FROM (the distributional hypothesis: "a word is known by the company it keeps")
    count co-occurrences in lots of unlabelled text -> PPMI -> SVD   (this lab, from scratch)
    word2vec / GloVe: the same idea, learned with a small neural network
    modern sentence embeddings: transformers trained on billions of sentences (Module 7)

THE DATA: an HR intranet corpus (6,000 unlabelled sentences) to LEARN from, and the HR knowledge base
    with 96 employee questions (gold article known) to TEST retrieval on. All offline.

LEVELS
    EASY   : One-hot, bag-of-words and TF-IDF by hand; the vocabulary-mismatch problem.
    MEDIUM : Learn word embeddings (PPMI + SVD) and LSA from unlabelled text; retrieval: TF-IDF vs
             LSA vs embeddings vs a hybrid.
    HARD   : Embeddings of anything (movies from ratings); the limits: word order and negation,
             unknown words, domain shift, dimension choice.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ml_lab_utils import (GENRES, HAS_MPL, banner, make_knowledge_base, make_ratings, make_text_corpus, note,
                          optional_import, plt, save_fig, show, step)
from unsup_toolkit import ALSMatrixFactorization, PPMIEmbeddings, cosine_sim, retrieval_report, tokenize

# %% =========================================================================
# LEVEL 1 (EASY): Sparse vectors, by hand
# ============================================================================
banner("From words to vectors: one-hot, bag-of-words, TF-IDF", "EASY")
sents = ["How many vacation days do I get", "Employees receive 25 days of paid holiday", "Annual time off allowance",
         "How do I reset my password"]
bow = CountVectorizer(stop_words="english")
B = bow.fit_transform(sents).toarray()
print(pd.DataFrame(B, columns=bow.get_feature_names_out(), index=[s[:30] for s in sents]).to_string())
tf = TfidfVectorizer(stop_words="english").fit(sents)
T = tf.transform(sents).toarray()
sim = pd.DataFrame([[cosine_sim(T[i], T[j]) for j in range(4)] for i in range(4)], index=[s[:30] for s in sents],
                   columns=["q1", "s2", "s3", "s4"])
print("\n  Cosine similarity of the TF-IDF vectors:")
print(sim.round(2).to_string())
note("""Each column is a word of the vocabulary; each sentence becomes a long, mostly-zero
vector of counts (TF-IDF additionally down-weights words that appear
everywhere). The question 'how many vacation days do I get' and the policy
sentence 'employees receive 25 days of paid holiday' share only the word
'days', and 'annual time off allowance' shares NOTHING with it: cosine 0. To a
sparse vector, 'vacation', 'holiday' and 'time off' are as different as
'vacation' and 'password'. This is the vocabulary-mismatch problem.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Learn dense vectors from unlabelled text
# ============================================================================
banner("Learning embeddings from 6,000 unlabelled intranet sentences", "MEDIUM")
corpus = make_text_corpus("hr", n_sentences=6000, seed=1)
print("  A few corpus sentences:", corpus[:3])
emb = PPMIEmbeddings(dim=50, window=4).fit(corpus)
show("Vocabulary learned / dimensions per word", f"{len(emb.vocab_)} / {emb.vectors_.shape[1]}")
for w in ["vacation", "salary", "supervisor", "doctor", "claim", "remote"]:
    show(f"most similar to '{w}'", ", ".join(f"{x} ({s:.2f})" for x, s in emb.most_similar(w, 4)))
note("""No labels, no dictionary: the algorithm only counted which words appear near
which other words, re-weighted the counts (PPMI: 'more often than chance?')
and compressed them with SVD to 50 numbers per word. Words used in the same
CONTEXTS ended up with similar vectors: vacation ~ leave ~ holiday, salary ~
wages ~ compensation, supervisor ~ manager. That is an embedding.""")

step("2.1  Retrieval on 96 employee questions: four representations")
articles, questions = make_knowledge_base("hr", seed=2)
docs = (articles.title + ". " + articles.text).tolist()
ids = articles.article_id.to_numpy()
gold = questions.gold_article.tolist()


def rank(S):
    return [list(ids[np.argsort(-row)]) for row in S]


tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(corpus + docs)
D_t, Q_t = tfidf.transform(docs), tfidf.transform(questions.question)
S_tfidf = (Q_t @ D_t.T).toarray()
svd = TruncatedSVD(100, random_state=0).fit(tfidf.transform(corpus + docs))   # LSA: SVD of the TF-IDF matrix
D_l = svd.transform(D_t)
Q_l = svd.transform(Q_t)
D_l /= np.linalg.norm(D_l, axis=1, keepdims=True) + 1e-12
Q_l /= np.linalg.norm(Q_l, axis=1, keepdims=True) + 1e-12
S_lsa = Q_l @ D_l.T
idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
D_e, Q_e = emb.embed(docs, idf), emb.embed(questions.question.tolist(), idf)
S_emb = Q_e @ D_e.T
S_hyb = 0.5 * S_tfidf / (S_tfidf.max(1, keepdims=True) + 1e-12) + 0.5 * S_emb
methods = {"TF-IDF (sparse, exact words)": S_tfidf, "LSA (SVD of TF-IDF, 100 dims)": S_lsa,
           "PPMI word embeddings (50 dims, idf-weighted average)": S_emb, "hybrid: TF-IDF + embeddings": S_hyb}
ret = pd.DataFrame({name: retrieval_report(rank(S), gold) for name, S in methods.items()}).T
print(ret.round(3).to_string())
fails = [i for i in range(len(gold)) if rank(S_tfidf[[i]])[0][0] != gold[i] and rank(S_emb[[i]])[0][0] == gold[i]]
if fails:
    i = fails[0]
    gold_title = articles.set_index("article_id").title[gold[i]]
    print(f"\n  Example: '{questions.question[i]}'")
    print(f"    TF-IDF's top answer: {articles.set_index('article_id').title[rank(S_tfidf[[i]])[0][0]]}   "
          f"embeddings' top answer: {gold_title} (correct)")
note(f"""Same 32 articles, same 96 questions. TF-IDF finds the right article first only
{ret.iloc[0, 0]:.0%} of the time: when the question says 'holiday' and the policy says
'leave', the words don't match. The learned embeddings bridge synonyms and
reach {ret.iloc[2, 0]:.0%}. LSA (the classic 1990 'latent semantic' method: SVD of the
TF-IDF matrix) barely changes the result here ({ret.iloc[1, 0]:.0%}). The 50/50 hybrid lands in
between ({ret.iloc[3, 0]:.0%}): on questions that are pure paraphrases, the exact-word half
drags it down. Hybrids win in production because real queries also contain
names, codes and numbers that only exact matching finds, but the blend weight
must be TUNED on your own questions (Labs 10 and 11 do this).""")

# %% =========================================================================
# LEVEL 3 (HARD): Embeddings of anything, and their limits
# ============================================================================
banner("Embeddings beyond words, and where they break", "HARD")
step("3.1  Movies as vectors, learned from ratings alone")
ratings, movies, _ = make_ratings(n_users=800, n_items=400, seed=7)
als = ALSMatrixFactorization(factors=8, reg=5.0, n_iter=15).fit(ratings, 800, 400)
Qv = als.Q_ / (np.linalg.norm(als.Q_, axis=1, keepdims=True) + 1e-12)
rated = np.bincount(ratings.item, minlength=400) >= 20
genre_sets = [set(g.split("|")) for g in movies.genres]
share_nn, share_rand = [], []
rng = np.random.default_rng(0)
for i in np.flatnonzero(rated):
    sims = Qv @ Qv[i]
    sims[i] = -np.inf
    sims[~rated] = -np.inf
    nn = int(np.argmax(sims))
    share_nn.append(bool(genre_sets[i] & genre_sets[nn]))
    share_rand.append(bool(genre_sets[i] & genre_sets[rng.choice(np.flatnonzero(rated))]))
show("Movies whose nearest neighbour (in the learned space) shares a genre", f"{np.mean(share_nn):.0%}")
show("... a random other movie shares a genre", f"{np.mean(share_rand):.0%}")
note("""Matrix factorisation (Lab 07) learned 8 numbers per movie from ratings alone.
Movies close in that space share a genre far more often than chance, although
the model never saw a genre. Users, products, songs, places: anything with
enough interactions can be embedded, and then 'similar' becomes cosine
similarity. Product embeddings power 'more like this' (Lab 09).""")

step("3.2  Where averaged word embeddings break")
pairs = [("parental leave is paid in full", "parental leave is not paid"),
         ("the employee pays the manager", "the manager pays the employee")]
for a, b in pairs:
    va, vb = emb.embed([a])[0], emb.embed([b])[0]
    show(f"'{a}' vs '{b}'", f"cosine {va @ vb:.2f}")
unknown = "What is the policy on sabbaticals?"
show("Words of a new question the embedding has never seen", [w for w in tokenize(unknown) if w not in emb.index_])
note("""Averaging word vectors throws away word ORDER and treats 'not' as just another
word: opposite statements look almost identical, and so do 'A pays B' and 'B
pays A'. Words outside the training vocabulary ('sabbaticals') get no vector at
all. Transformer models (Module 7) fix both: they read the whole sentence, in
order, and split unknown words into known pieces.""")

step("3.3  Domain shift and the number of dimensions")
it_articles, it_questions = make_knowledge_base("it", seed=3)
it_docs = (it_articles.title + ". " + it_articles.text).tolist()
it_ids = it_articles.article_id.to_numpy()
it_tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(it_docs)
S_it_t = (it_tfidf.transform(it_questions.question) @ it_tfidf.transform(it_docs).T).toarray()
rows = {"IT questions, TF-IDF": retrieval_report([list(it_ids[np.argsort(-r)]) for r in S_it_t], it_questions.gold_article.tolist())}
S_it_hr = emb.embed(it_questions.question.tolist()) @ emb.embed(it_docs).T
rows["IT questions, embeddings learned on HR text"] = retrieval_report([list(it_ids[np.argsort(-r)]) for r in S_it_hr],
                                                                      it_questions.gold_article.tolist())
emb_it = PPMIEmbeddings(dim=50).fit(make_text_corpus("it", n_sentences=6000, seed=1))
S_it_it = emb_it.embed(it_questions.question.tolist()) @ emb_it.embed(it_docs).T
rows["IT questions, embeddings learned on IT text"] = retrieval_report([list(it_ids[np.argsort(-r)]) for r in S_it_it],
                                                                      it_questions.gold_article.tolist())
print(pd.DataFrame(rows).T.round(3).to_string())
dims = {}
for d in [5, 20, 50, 150]:
    e_d = PPMIEmbeddings(dim=d).fit(corpus)
    dims[d] = retrieval_report(rank(e_d.embed(questions.question.tolist(), idf) @ e_d.embed(docs, idf).T), gold)["recall@1"]
print(pd.Series(dims, name="HR recall@1 by embedding dimensions").round(3).to_string())
st = optional_import("sentence_transformers")
if st is not None:
    model = st.SentenceTransformer("all-MiniLM-L6-v2")
    Ds, Qs = model.encode(docs, normalize_embeddings=True), model.encode(questions.question.tolist(), normalize_embeddings=True)
    show("A pretrained sentence-transformer (installed), HR recall@1", f"{retrieval_report(rank(Qs @ Ds.T), gold)['recall@1']:.3f}")
note(f"""Embeddings know only the language they were trained on. HR embeddings applied
to IT questions ('VPN', 'passcode') have most words missing or misplaced and
do worse than plain TF-IDF; embeddings trained on IT text recover. Too few
dimensions blur distinct meanings together (5 dimensions: {dims[5]:.0%} recall@1);
here more helps up to 150, with diminishing gains.
That's why practitioners pick an embedding model trained on similar text,
and evaluate retrieval on their OWN questions before trusting it. (With the
`sentence-transformers` package installed, this lab adds a pretrained model.)""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    words = ["vacation", "holiday", "leave", "salary", "wages", "pay", "manager", "supervisor", "doctor", "physician",
             "claim", "reimbursement", "pension", "overtime"]
    words = [w for w in words if w in emb.index_]
    V = emb.vectors_[[emb.index_[w] for w in words]]
    P = V @ np.linalg.svd(V - V.mean(0), full_matrices=False)[2][:2].T
    axes[0].scatter(P[:, 0], P[:, 1])
    for w, (x_, y_) in zip(words, P):
        axes[0].annotate(w, (x_, y_), fontsize=9, xytext=(3, 3), textcoords="offset points")
    axes[0].set(title="Learned word vectors, projected to 2-D (PCA)")
    r = ret["recall@1"]
    axes[1].barh(r.index, r.values, color=["#9d9d9d", "#4c78a8", "#54a24b", "#f58518"])
    axes[1].set(xlim=(0, 1), xlabel="recall@1 on 96 HR questions", title="Retrieval: sparse vs learned vectors")
    axes[1].tick_params(axis="y", labelsize=8)
    save_fig(fig, "08_embeddings")

banner("KEY TAKEAWAYS")
print("""  1. Sparse vectors (bag-of-words, TF-IDF) match exact words: synonyms look unrelated (vocabulary mismatch).
  2. Embeddings are dense vectors learned from context; similar usage -> similar vectors (PPMI+SVD, word2vec).
  3. Learned embeddings bridge synonyms: retrieval improves; a hybrid with exact matching is best.
  4. Anything with interactions can be embedded: users, movies, products (matrix factorisation).
  5. Averaged word vectors ignore order and negation; embeddings break across domains: evaluate on your data.""")
