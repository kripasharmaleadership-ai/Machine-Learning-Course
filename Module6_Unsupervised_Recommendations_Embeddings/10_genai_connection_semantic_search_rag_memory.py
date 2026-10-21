"""
LAB 10 - Connection to GenAI: Text Embeddings, Semantic Search, Vector Databases, RAG and Agent Memory
======================================================================================================

THIS MODULE IS THE DIRECT BRIDGE TO RETRIEVAL-AUGMENTED GENERATION (RAG)
    1. TEXT EMBEDDINGS     every document chunk and every question becomes a vector       (Lab 08)
    2. SEMANTIC SEARCH     the answer lives in the chunks whose vectors are closest       (cosine, Lab 02)
    3. VECTOR DATABASE     an index that finds the nearest vectors among millions, fast   (k-means cells, Lab 03)
    4. DOCUMENT RETRIEVAL  chunking, metadata filters, the top-k passages                 (this lab)
    5. RAG                 retrieved passages + the question -> a prompt -> an LLM answers WITH citations
    6. AGENT MEMORY        an assistant stores facts as vectors and recalls the relevant ones later

    No LLM or API key is needed here: the 'generation' step is extractive (return the best supporting
    sentence with its source). The PROMPT you would send to an LLM is built and printed, and everything
    before it (where most RAG systems fail) is real and measured.

LEVELS
    EASY   : Semantic search on the HR knowledge base: embeddings vs keywords.
    MEDIUM : A vector database from scratch: exact vs IVF index at 30,000 vectors; chunking for retrieval.
    HARD   : A RAG pipeline end to end: prompt with citations, grounded answers, 'I don't know';
             stale documents and metadata filters; agent memory with recency.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_lab_utils import (HAS_MPL, banner, make_knowledge_base, make_product_catalogue, make_text_corpus, note, plt, save_fig,
                          show, step)
from unsup_toolkit import (AgentMemory, FlatIndex, IVFIndex, PPMIEmbeddings, chunk_text, extractive_answer,
                           retrieval_report)

articles, questions = make_knowledge_base("hr", seed=10)
corpus = make_text_corpus("hr", n_sentences=6000, seed=1)
emb = PPMIEmbeddings(dim=60).fit(corpus)
tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(corpus + articles.text.tolist())
IDF = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))


def embed(texts):
    return emb.embed(list(texts), IDF)


docs = (articles.title + ". " + articles.text).tolist()
ids = articles.article_id.to_numpy()

# %% =========================================================================
# LEVEL 1 (EASY): Semantic search
# ============================================================================
banner("Semantic search over the HR handbook", "EASY")
index = FlatIndex().add(embed(docs), ids)
title = articles.set_index("article_id").title
for q in ["how much holiday do I get each year", "my supervisor wants me to work extra hours, how is that paid",
          "do I need a GP note when I'm ill"]:
    hits = index.search(embed([q]), k=3)
    kw = (tfidf.transform([q]) @ tfidf.transform(docs).T).toarray()[0]
    print(f"\n  Q: {q}")
    print("    semantic : " + " | ".join(f"{title[i]} ({s:.2f})" for i, s in hits))
    print("    keyword  : " + " | ".join(f"{title[ids[j]]} ({kw[j]:.2f})" for j in np.argsort(-kw)[:3]))
sem = retrieval_report([[i for i, _ in index.search(v, k=len(ids))] for v in embed(questions.question)], questions.gold_article.tolist())
show("Semantic search over 96 questions: recall@1 / recall@3", f"{sem['recall@1']:.0%} / {sem['recall@3']:.0%}")
note("""This is the 'R' of RAG. The question is embedded with the SAME model as the
documents, and the nearest vectors are the most relevant pages, whatever words
the employee used ('holiday' -> Annual leave; 'GP note' -> Sick leave), where
keyword search is distracted by shared words ('holiday' -> Holiday party). It
is not perfect: for 'supervisor ... extra hours' Overtime is only second,
behind a page that shares the words 'hours' and 'work'. Everything a RAG system can answer depends on this step.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): A vector database, and chunking
# ============================================================================
banner("A vector database from scratch, and chunking documents", "MEDIUM")
step("2.1  30,000 product vectors: exact search vs an IVF index")
shop = make_product_catalogue(per_sub=3000, seed=21)
shop_text = (shop.title + ". " + shop.description).str.lower()
p_emb = PPMIEmbeddings(dim=60).fit(make_product_catalogue(per_sub=300, seed=99, nouns_per_product=2).pipe(
    lambda d: (d.title + ". " + d.description).str.lower().tolist()))
t0 = time.perf_counter()
P = p_emb.embed(shop_text.tolist())
show("Products embedded / seconds", f"{len(P):,} / {time.perf_counter() - t0:.1f}")
queries = p_emb.embed(["wireless earbuds", "velvet couch", "espresso machine", "camping shelter", "fitness tracker",
                       "dimmable desk lamp", "pilates mat", "smoothie maker", "trail sneakers", "lightweight ultrabook"] * 10)
flat = FlatIndex().add(P, shop.product_id)
truth = [set(i for i, _ in flat.search(q, k=10)) for q in queries]
rows = {}
t0 = time.perf_counter()
for q in queries:
    flat.search(q, k=10)
rows["flat (exact, compares every vector)"] = {"vectors compared per query": len(P), "recall@10 vs exact": 1.0,
                                               "ms per query": (time.perf_counter() - t0) / len(queries) * 1000}
t0 = time.perf_counter()
ivf = IVFIndex(n_lists=64, nprobe=1).add(P, shop.product_id)
show("IVF index build (k-means into 64 cells), seconds", f"{time.perf_counter() - t0:.1f}")
for nprobe in [1, 4, 16]:
    ivf.nprobe = nprobe
    t0 = time.perf_counter()
    found = [set(i for i, _ in ivf.search(q, k=10)) for q in queries]
    ms = (time.perf_counter() - t0) / len(queries) * 1000
    rows[f"IVF, 64 cells, search {nprobe} nearest cell(s)"] = {"vectors compared per query": ivf.last_compared_,
                                                               "recall@10 vs exact": np.mean([len(a & b) / 10 for a, b in zip(found, truth)]),
                                                               "ms per query": ms}
vdb = pd.DataFrame(rows).T
print(vdb.round(3).to_string())
note("""A vector database answers 'which stored vectors are nearest to this one?'. The
exact way compares the query with every vector: fine for thousands, too slow
for hundreds of millions. An IVF (inverted file) index first clusters the
vectors with k-means (Lab 03) and stores each cell's members; a query is
compared only with the members of its nearest cell(s). Searching 1 of 64 cells
compares ~1/64 of the vectors; probing more cells buys back recall. FAISS,
Milvus, pgvector and most vector databases offer this (plus HNSW graphs).""")

step("2.2  Chunking: retrieve the passage, not the whole document")
BOILERPLATE = ["This page is part of the employee handbook.", "It is owned by the people team and reviewed every year.",
               "Printed copies may be out of date, so always check the intranet version.",
               "Managers should share this page with new starters during their first week.",
               "Translations are available on request from the people team.",
               "Where local law gives employees better terms, local law applies.",
               "Questions about this page can be raised through the HR portal.",
               "Previous versions of this page are archived and no longer apply.",
               "The glossary explains any terms used on this page.",
               "Feedback on the clarity of this page is always welcome."]
rng = np.random.default_rng(0)
long_docs = []
for _, a in articles.iterrows():                                      # make articles longer, like real policy pages
    pre, post = (" ".join(rng.choice(BOILERPLATE, 5, replace=False)) for _ in range(2))
    long_docs.append((a.article_id, f"{a.title}. {pre} {a.text} {post}", a.answer_fact))
BUDGET = 80                                                           # words of context we can afford per question


def context_within_budget(top, budget=BUDGET):
    """Concatenate the retrieved texts in rank order and cut at the word budget (what the LLM actually sees)."""
    return " ".join(" ".join(c for _, c in top).split()[:budget])


rows = {}
for size in [12, 25, 50, 10_000]:
    chunks = [(aid, c) for aid, text, _ in long_docs for c in chunk_text(text, max_words=size, overlap=size // 4 if size < 1000 else 0)]
    cidx = FlatIndex().add(embed([c for _, c in chunks]), range(len(chunks)))
    hit, hit_budget = [], []
    for q, g, ans in zip(questions.question, questions.gold_article, questions.answer):
        top = [chunks[i] for i, _ in cidx.search(embed([q]), k=3)]
        hit.append(any(aid == g and ans.lower() in c.lower() for aid, c in top))
        hit_budget.append(ans.lower() in context_within_budget(top).lower())
    rows["whole documents" if size >= 1000 else f"chunks of ~{size} words"] = {
        "chunks stored": len(chunks), "fact in top-3 texts (unlimited context)": np.mean(hit),
        f"fact in the first {BUDGET} words of context": np.mean(hit_budget),
        "words in top-3 texts": int(np.mean([len(c.split()) for _, c in chunks]) * 3)}
chunk_tab = pd.DataFrame(rows).T
print(chunk_tab.round(3).to_string())
BCOL = f"fact in the first {BUDGET} words of context"
best_budget = chunk_tab[BCOL].idxmax()
note(f"""Two forces pull in opposite directions. With unlimited context, whole pages
reach {chunk_tab.loc['whole documents'].iloc[1]:.0%} of answer facts: each page is about one topic, so the page
vector matches well and the fact is always somewhere inside. But the LLM must
then read ~{int(chunk_tab.loc['whole documents'].iloc[3])} words per question, which costs money, adds latency and
buries the fact in boilerplate. Under a budget of {BUDGET} words the winner is
'{best_budget}' ({chunk_tab[BCOL].max():.0%} vs {chunk_tab.loc['whole documents', BCOL]:.0%} for whole pages): small enough
that the fact sits near the top, big enough to keep its context. Chunk size
(and overlap) is a hyperparameter of a RAG system: tune it on questions with
known answers, under your real context budget, exactly like this table.""")

# %% =========================================================================
# LEVEL 3 (HARD): RAG end to end, stale documents, agent memory
# ============================================================================
banner("Retrieval-augmented generation, end to end", "HARD")
chunks = [(aid, c) for aid, text, _ in long_docs for c in chunk_text(text, max_words=25, overlap=8)]
# (~50-word chunks retrieve a little more, but boilerplate dilutes their scores: with the same "I don't know"
#  threshold they wrongly refuse 25% of real questions instead of 11%. Chunk size and threshold are tuned together.)
cidx = FlatIndex().add(embed([c for _, c in chunks]), range(len(chunks)))


def rag(question, k=3, min_score=0.55, filter_fn=None):
    hits = [(chunks[i], s) for i, s in cidx.search(embed([question]), k=k * 4)]
    hits = [(c, s) for c, s in hits if filter_fn is None or filter_fn(c[0])][:k]
    if not hits or hits[0][1] < min_score:
        return {"answer": "I don't know: the handbook doesn't cover this. Please contact HR.", "source": None,
                "score": hits[0][1] if hits else 0.0, "prompt": None}
    context = "\n".join(f"[{c[0]}] {c[1]}" for c, _ in hits)
    prompt = (f"Answer the employee's question using ONLY the sources below. Cite the source id. If the sources don't "
              f"contain the answer, say you don't know.\n\nSOURCES:\n{context}\n\nQUESTION: {question}\nANSWER:")
    out = extractive_answer(question, [(c[0], c[1]) for c, _ in hits], embed)
    out["prompt"], out["score"] = prompt, hits[0][1]
    return out


step("3.1  One question through the whole pipeline")
demo = rag("how many paid days of vacation does a staff member get")
print(demo["prompt"])
print(f"\n  EXTRACTIVE ANSWER: {demo['answer']}  [source {demo['source']}]")
note("""Retrieve the top chunks, put them in a prompt with their ids, instruct the model to
answer ONLY from them and to cite, and allow 'I don't know'. With an LLM API
you would send exactly this prompt; here an extractive step picks the
sentence that best answers the question. Grounding the answer in retrieved
text, with a citation, is what makes RAG checkable.""")

step("3.2  Evaluate answers, and refuse what the handbook doesn't cover")
oos = ["what is the CEO's salary", "can I bring my dog to the office", "how do I reset my laptop password",
       "what is the wifi password for guests", "when is the next board meeting", "how do I file my personal taxes"]
rows = []
for q, ans, g in zip(questions.question, questions.answer, questions.gold_article):
    r = rag(q)
    rows.append({"question": q, "in scope": True, "answered": r["source"] is not None,
                 "fact in the prompt": r["prompt"] is not None and ans.lower() in r["prompt"].lower(),
                 "fact in the extractive answer": ans.lower() in r["answer"].lower(),
                 "cited the right page": r["source"] == g, "top score": r["score"], "answer": r["answer"], "fact": ans})
for q in oos:
    r = rag(q)
    rows.append({"question": q, "in scope": False, "answered": r["source"] is not None, "top score": r["score"],
                 "answer": r["answer"]})
ev = pd.DataFrame(rows)
ins, out = ev[ev["in scope"] == True], ev[ev["in scope"] == False]  # noqa: E712
summary = pd.Series({"in-scope: answer fact reached the prompt (an LLM could answer)": ins["fact in the prompt"].mean(),
                     "in-scope: one-sentence extractive answer contains the fact": ins["fact in the extractive answer"].mean(),
                     "in-scope: cited the right page": ins["cited the right page"].mean(),
                     "in-scope questions wrongly refused": 1 - ins.answered.mean(),
                     "out-of-scope questions correctly refused": 1 - out.answered.mean()})
print(summary.round(3).to_string())
show("Top retrieval score: in-scope median / out-of-scope median",
     f"{ins['top score'].median():.2f} / {out['top score'].median():.2f}")
miss = ins[ins["cited the right page"].astype(bool) & ~ins["fact in the extractive answer"].astype(bool)].head(3)
print("\n  Right page, but the graded fact is not in the chosen sentence:")
for _, m in miss.iterrows():
    print(f"    Q: {m.question!r}  -> {m.answer!r}   (graded fact: {m.fact!r})")
leaks = out[out.answered]
for _, m in leaks.iterrows():
    print(f"  Out of scope but ANSWERED (score {m['top score']:.2f}): {m.question!r} -> {m.answer!r}")
leak_note = "The leaks above share words with real pages: a threshold alone is not a complete guard." if len(leaks) else ""
note(f"""Evaluate a RAG system in stages, because each stage fails differently.
RETRIEVAL: the fact reached the prompt for {summary.iloc[0]:.0%} of questions; that is the
ceiling for any LLM. GENERATION: our one-sentence extractor states the fact
only {summary.iloc[1]:.0%} of the time. Some misses are real (the best-matching sentence is
not the one with the number); others are the EVALUATION's fault: each page has
one graded fact, so a correct answer to 'does the budget cover exams?' is
marked wrong. An LLM reading the whole prompt does much better, and several
reference answers per question (or an LLM-as-judge) grade more fairly.
REFUSAL: out-of-scope questions retrieve weak matches (median top score
{out['top score'].median():.2f} vs {ins['top score'].median():.2f}), so a minimum-similarity threshold says 'I don't know'
for {summary.iloc[4]:.0%} of them while wrongly refusing {summary.iloc[3]:.0%} of real questions. {leak_note}
The threshold is the precision/recall trade-off of Module 4, and the main
defence against confident answers from irrelevant pages.""")

step("3.3  Stale documents: two versions of the same policy")
old = ("HR-00-2019", "Annual leave allowance (2019 version). Every employee receives 20 days of paid leave per year. "
       "Unused leave expires at the end of the year.")
chunks.append(old)
cidx = FlatIndex().add(embed([c for _, c in chunks]), range(len(chunks)))
q = "how many days of paid leave per year"
r_any = rag(q)
r_cur = rag(q, filter_fn=lambda source: not source.endswith("2019"))
print(f"  without a version filter: {r_any['answer']}  [source {r_any['source']}]")
print(f"  current documents only  : {r_cur['answer']}  [source {r_cur['source']}]")
note("""An outdated copy of a policy is just as similar to the question as the current
one, so retrieval can return it and the system answers with yesterday's rule,
confidently and with a citation. Retrieval quality is not only about vectors:
it needs METADATA (version, date, department, access rights) stored next to
each vector and used as a FILTER, plus a process to remove superseded documents.""")

step("3.4  Agent memory: remembering what the user told the assistant")
memory = AgentMemory(embed=embed, recency_weight=0.0)
log = [(1, "I work in the Lisbon office"), (2, "my manager is Ana from the finance team"),
       (3, "I am planning parental leave from March"), (4, "I prefer answers by email"),
       (9, "I now work in the Dublin office")]
for t, fact in log:
    memory.remember(fact, t)
for weight in [0.0, 0.5]:
    memory.recency_weight = weight
    top = memory.recall("which office do I work in", now=10, k=1)[0][0]
    show(f"'which office do I work in?' (recency weight {weight})", top)
show("'who is my manager?' (recency weight 0.5)", memory.recall("who is my manager", now=10, k=1)[0][0])
note("""An agent's long-term memory is a small vector database of things the user said,
recalled by similarity to the current question. Pure similarity can't tell an
old fact from its update: 'Lisbon' and 'Dublin' are equally similar to 'which
office'. Weighting by recency (or storing facts with keys and overwriting
them) recalls the current one, while a strongly relevant old fact (the
manager, said at turn 2) still wins over recent but irrelevant ones. Memory needs the same care as RAG: relevance,
freshness, and deleting what is no longer true.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    v = vdb.iloc[1:]
    axes[0].plot(v["vectors compared per query"], v["recall@10 vs exact"], "o-")
    for n_, r_ in v.iterrows():
        axes[0].annotate(n_.split("search ")[1], (r_["vectors compared per query"], r_["recall@10 vs exact"]), fontsize=8,
                         xytext=(4, -10), textcoords="offset points")
    axes[0].set(xscale="log", xlabel="vectors compared per query", ylabel="recall@10 vs exact search", title="IVF: speed vs recall")
    c = chunk_tab[BCOL]
    axes[1].bar(c.index, c.values, color="#4c78a8")
    axes[1].set(ylim=(0, 1), title=f"Answer fact within a {BUDGET}-word context")
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[2].hist(ev[ev["in scope"]]["top score"], bins=15, alpha=0.6, label="in-scope questions")
    axes[2].hist(ev[~ev["in scope"]]["top score"], bins=6, alpha=0.6, label="out-of-scope questions")
    axes[2].axvline(0.55, color="r", ls="--", label="'I don't know' threshold")
    axes[2].set(xlabel="top retrieval similarity", title="Knowing when not to answer")
    axes[2].legend(fontsize=8)
    save_fig(fig, "10_rag_vector_db")

banner("KEY TAKEAWAYS")
print("""  1. RAG = embed chunks + embed the question + nearest-neighbour search + a grounded, cited answer.
  2. Vector databases trade a little recall for huge speed: IVF searches only the nearest k-means cells.
  3. Chunk size is a hyperparameter: tune it on questions with known answers.
  4. A similarity threshold lets the system say 'I don't know'; metadata filters keep stale documents out.
  5. Agent memory is a vector store of user facts: retrieve by relevance AND recency.""")
