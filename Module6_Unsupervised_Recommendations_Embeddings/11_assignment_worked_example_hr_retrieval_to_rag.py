"""
LAB 11 - Assignment Worked Example: An HR Document-Similarity Prototype, and the Path to a RAG Application
==========================================================================================================

THE BRIEF (from the People team)
    "Employees ask the HR inbox the same questions every day. Build a prototype that, given a question,
     finds the right handbook page. If it works, we want to turn it into an assistant that answers
     with a citation (RAG). Tell us which approach to use and what it would take."

    Requirements agreed with the People team:
        HARD  recall@3 >= 0.90 on held-out questions   (the right page in the top 3 for 9 questions in 10)
        HARD  <= 20 ms per query                         (it sits behind a chat box)
        soft  recall@1 (40%), MRR (20%), refusing out-of-scope questions (30%), small index (10%)

    This lab is the model answer to ASSIGNMENT.md, done on the HR handbook. The assignment asks you to do
    the same for the IT help centre, so read this as a template, not as your answer.

LEVELS
    EASY   : Data, a dev/test split of the questions, and five candidate retrievers.
    MEDIUM : Tune on dev (hybrid weight, refusal threshold), score on test, compare with the scorecard.
    HARD   : Error analysis, and a design memo: how this prototype becomes a RAG application.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_lab_utils import (HAS_MPL, OUT_DIR, banner, make_knowledge_base, make_text_corpus, note, plt, save_fig, show,
                          step)
from unsup_toolkit import ModelScorecard, PPMIEmbeddings, Requirement, chunk_text, df_to_md, retrieval_report

# %% =========================================================================
# LEVEL 1 (EASY): Data and candidates
# ============================================================================
banner("The data, the split and the candidates", "EASY")
articles, questions = make_knowledge_base("hr", seed=11)
corpus = make_text_corpus("hr", n_sentences=6000, seed=2)              # unlabelled HR text for the embeddings
show("Handbook pages (incl. distractor pages) / questions", f"{len(articles)} / {len(questions)}")
print(articles[["article_id", "title"]].head(6).to_string(index=False))

# split questions BY QUESTION, stratified by page: every page has dev and test questions
rng = np.random.default_rng(0)
questions["split"] = "test"
for _, grp in questions.groupby("gold_article"):
    questions.loc[rng.choice(grp.index, len(grp) // 2, replace=False), "split"] = "dev"
dev, test = questions[questions.split == "dev"], questions[questions.split == "test"]
OOS = ["what is the CEO's salary", "can I bring my dog to work", "how do I reset my laptop password",
       "what is the guest wifi password", "when is the next board meeting", "how do I file my personal taxes",
       "which printer is closest to me", "where can I buy a train ticket", "is the canteen vegetarian friendly",
       "how do I install the VPN", "what is the share price today", "who won the football match"]
oos_dev, oos_test = OOS[::2], OOS[1::2]
show("Dev / test questions; out-of-scope dev / test", f"{len(dev)} / {len(test)}; {len(oos_dev)} / {len(oos_test)}")
note("""The labelled questions are split in two. DEV is for every decision (hybrid
weight, refusal threshold, chunk size); TEST is touched once, at the end, for
the scorecard. Tuning on the same questions you report would overstate the
prototype, the oldest mistake in ML (Module 2) and very common in RAG demos.
Out-of-scope questions are needed too: an assistant must also know when the
handbook has no answer.""")

docs = (articles.title + ". " + articles.text).tolist()
ids = articles.article_id.to_numpy()
emb = PPMIEmbeddings(dim=60).fit(corpus)
word_tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(corpus + docs)
char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(docs)
IDF = dict(zip(word_tfidf.get_feature_names_out(), word_tfidf.idf_))
D_word, D_char, D_emb = word_tfidf.transform(docs), char_tfidf.transform(docs), emb.embed(docs, IDF)
chunks = [(aid, c) for aid, d in zip(ids, docs) for c in chunk_text(d, max_words=20, overlap=6)]
chunk_owner = np.array([aid for aid, _ in chunks])
C_emb, C_word = emb.embed([c for _, c in chunks], IDF), word_tfidf.transform([c for _, c in chunks])


def page_scores(qs, kind, alpha=0.5):
    """Similarity of every question to every page (n_questions x n_pages)."""
    if kind == "word TF-IDF":
        return (word_tfidf.transform(qs) @ D_word.T).toarray()
    if kind == "char TF-IDF":
        return (char_tfidf.transform(qs) @ D_char.T).toarray()
    if kind == "embeddings":
        return emb.embed(qs, IDF) @ D_emb.T
    if kind == "hybrid":
        return alpha * (emb.embed(qs, IDF) @ D_emb.T) + (1 - alpha) * (word_tfidf.transform(qs) @ D_word.T).toarray()
    if kind == "hybrid on chunks":                                    # score a page by its best-matching chunk
        s = alpha * (emb.embed(qs, IDF) @ C_emb.T) + (1 - alpha) * (word_tfidf.transform(qs) @ C_word.T).toarray()
        return np.column_stack([s[:, chunk_owner == a].max(axis=1) for a in ids])
    raise ValueError(kind)


def evaluate(qs, gold, kind, alpha=0.5):
    S = page_scores(list(qs), kind, alpha)
    return retrieval_report([list(ids[np.argsort(-row)]) for row in S], list(gold)), S


CANDIDATES = ["word TF-IDF", "char TF-IDF", "embeddings", "hybrid", "hybrid on chunks"]
print("\n  candidates: " + " | ".join(CANDIDATES))

# %% =========================================================================
# LEVEL 2 (MEDIUM): Tune on dev, score on test
# ============================================================================
banner("Tune on dev, score once on test, compare with a scorecard", "MEDIUM")
step("2.1  Tune the hybrid weight on DEV")
grid = np.round(np.linspace(0, 1, 11), 1)
tune = pd.DataFrame({kind: [evaluate(dev.question, dev.gold_article, kind, a)[0]["MRR"] for a in grid]
                     for kind in ["hybrid", "hybrid on chunks"]}, index=grid)
tune.index.name = "alpha (weight on embeddings)"
print(tune.round(3).T.to_string())
ALPHA = {k: float(tune[k].idxmax()) for k in tune}
show("Best alpha on dev (hybrid / hybrid on chunks)", f"{ALPHA['hybrid']} / {ALPHA['hybrid on chunks']}")


def refusal_threshold(kind):
    """Chosen on DEV: the highest threshold that still answers 92% of in-scope dev questions."""
    top = page_scores(list(dev.question), kind, ALPHA.get(kind, 0.5)).max(axis=1)
    return float(np.quantile(top, 0.08))


step("2.2  Score every candidate on TEST")
card = ModelScorecard(
    problem="HR handbook page finder: which retrieval approach should the prototype (and later the RAG assistant) use?",
    requirements=[
        Requirement("right page in top 3", "recall@3", 0.90, higher_is_better=True, hard=True,
                    why="employees see three suggestions; the right page must be among them"),
        Requirement("latency", "ms_per_query", 20, hard=True, why="it sits behind a chat box"),
        Requirement("right page first", "recall@1", higher_is_better=True, weight=0.40, why="the RAG answer cites the top page"),
        Requirement("ranking quality", "MRR", higher_is_better=True, weight=0.20, why="the right page near the top"),
        Requirement("refuses out-of-scope", "oos_refused", higher_is_better=True, weight=0.30,
                    why="no confident answers from unrelated pages"),
        Requirement("index size", "index_kb", weight=0.10, why="cost of storing vectors for 10,000+ pages later")])
test_scores = {}
for kind in CANDIDATES:
    a = ALPHA.get(kind, 0.5)
    t0 = time.perf_counter()
    rep, S = evaluate(test.question, test.gold_article, kind, a)
    ms = (time.perf_counter() - t0) / len(test) * 1000
    th = refusal_threshold(kind)
    oos_top = page_scores(oos_test, kind, a).max(axis=1)
    index_kb = {"word TF-IDF": D_word.data.nbytes, "char TF-IDF": D_char.data.nbytes, "embeddings": D_emb.nbytes,
                "hybrid": D_emb.nbytes + D_word.data.nbytes, "hybrid on chunks": C_emb.nbytes + C_word.data.nbytes}[kind] / 1024
    test_scores[kind] = S
    card.add(kind, {"recall@1": rep["recall@1"], "recall@3": rep["recall@3"], "MRR": rep["MRR"], "ms_per_query": ms,
                    "oos_refused": float(np.mean(oos_top < th)), "in_scope_refused": float(np.mean(S.max(axis=1) < th)),
                    "index_kb": index_kb, "threshold": th},
             note=f"alpha={a}" if "hybrid" in kind else "")
table = card.evaluate()
print(table.drop(columns=["failed gates"]).round(3).to_string())
winner = card.recommend()
res = pd.DataFrame(card.results).T
show("Recommendation", winner)
failed = table[~table.eligible.astype(bool)]
hybrid_note = ("Tuning on dev put ALL the weight on embeddings (alpha = 1.0): on this vocabulary the keyword "
               "score adds nothing, the 'hybrid' IS the embedding model, and the scorecard prefers the simpler, "
               "smaller index. Keep the hybrid in the code anyway: on real text with product names, codes and "
               "numbers, keywords usually earn weight." if ALPHA["hybrid"] == 1.0 else
               f"The tuned hybrid (alpha = {ALPHA['hybrid']}) keeps the embeddings' recall while exact words still count.")
note(f"""On held-out questions, keyword methods miss paraphrases (word TF-IDF recall@3
{res.loc['word TF-IDF', 'recall@3']:.0%}, char TF-IDF {res.loc['char TF-IDF', 'recall@3']:.0%}); embeddings understand 'holiday' ~ 'leave'
(recall@3 {res.loc['embeddings', 'recall@3']:.0%}). Failing the recall@3 gate: {', '.join(failed.index) or 'nobody'}.
{hybrid_note}
Chunking HURT here (recall@3 {res.loc['hybrid on chunks', 'recall@3']:.0%}): these pages are only a few sentences long, so a
20-word chunk loses the title and context that made the page match. Chunk when
pages are long (Lab 10), not by reflex. The scorecard recommends '{winner}':
recall@1 {res.loc[winner, 'recall@1']:.0%}, recall@3 {res.loc[winner, 'recall@3']:.0%}, refusing {res.loc[winner, 'oos_refused']:.0%} of out-of-scope
questions at a threshold chosen on dev. Every candidate is far below the 20 ms
limit at this size; latency matters at millions of chunks, which is what the
vector index of Lab 10 is for.""")

# %% =========================================================================
# LEVEL 3 (HARD): Error analysis and the design memo
# ============================================================================
banner("Error analysis, and from prototype to RAG application", "HARD")
step("3.1  Where does the winner fail?")
S = test_scores[winner]
top1 = ids[S.argmax(axis=1)]
errs = test.assign(retrieved=top1, rank_of_gold=[int(np.where(ids[np.argsort(-row)] == g)[0][0]) + 1
                                                 for row, g in zip(S, test.gold_article)])
errs = errs[errs.retrieved != errs.gold_article]
title = articles.set_index("article_id").title
print(errs.assign(gold=errs.gold_article.map(title), got=errs.retrieved.map(title))[
    ["question", "gold", "got", "rank_of_gold"]].head(10).to_string(index=False))
confused = (errs.gold_article.map(title) + "  ->  " + errs.retrieved.map(title)).value_counts().head(5)
show("Most frequent confusions", "\n    " + "\n    ".join(f"{n}x  {pair}" for pair, n in confused.items()))
distractor_share = errs.retrieved.str.contains("X").mean() if len(errs) else 0.0
note(f"""{len(errs)} of {len(test)} test questions get the wrong page first; the right one is usually
still in the top 3 (median rank {errs.rank_of_gold.median() if len(errs) else 1:.0f}). {distractor_share:.0%} of the mistakes land on a DISTRACTOR page
(one that shares the vocabulary but answers a different question), the rest
on a related policy (leave types, pay topics). These are fixable: add synonyms
the embedding corpus lacks, write page titles that say what the page answers,
merge pages that overlap, or add a re-ranker (a model that re-scores the top
10). Error analysis tells you WHICH fix is worth building.""")

step("3.2  The design memo")
extra = f"""## Held-out evaluation

- {len(test)} test questions ({len(dev)} used for tuning), {len(oos_test)} out-of-scope test questions.
- The winner's hybrid weight was chosen on dev: alpha = {ALPHA.get(winner, 0.5)} (weight on embeddings).
- Most frequent confusions on test:

{chr(10).join(f'  - {n}x {pair}' for pair, n in confused.items())}

## From this prototype to a RAG application

| Stage | Prototype (this lab) | Production RAG |
|---|---|---|
| Ingestion | 32 synthetic pages | Connectors to the HR wiki / SharePoint; re-index on every page change |
| Chunking | whole pages (short); ~20-word chunks tested and rejected | Real pages are long: tune chunk size on dev questions under the real context budget (Lab 10) |
| Embeddings | PPMI + SVD trained on HR text | A pre-trained sentence-embedding model (keep the same dev and test sets to compare) |
| Vector store | numpy arrays, exact search | A vector database (pgvector, FAISS, a managed service) with IVF/HNSW indexes |
| Metadata | page id | version, effective date, country, audience, access rights: used as FILTERS |
| Retrieval | hybrid score, top 3 | Hybrid (BM25 + vectors) top 20, then a re-ranker, then top 3-5 |
| Generation | none | An LLM prompted to answer ONLY from the sources, cite page ids, say "I don't know" |
| Refusal | similarity threshold tuned on dev | Threshold + the LLM's own "not in the sources" + escalation to a human |
| Evaluation | recall@k, MRR, refusal rate | The same retrieval metrics plus answer correctness, citation accuracy, faithfulness |
| Monitoring | none | Log questions, citations and thumbs up/down; review refusals weekly; re-test on every model change |

## Risks to manage

1. **Stale or conflicting policies**: an old version is as similar as the current one; filter by version and date (Lab 10).
2. **Access control**: retrieval must never return pages the employee may not see (e.g., manager-only pay bands).
3. **Confident wrong answers**: keep citations visible and the refusal path tested with out-of-scope questions.
4. **Evaluation drift**: the {len(questions)} labelled questions are a start; add real (anonymised) questions from the inbox every month.
5. **Personal data**: questions can contain health or family details; set retention and redaction rules for the logs.
"""
memo = card.to_markdown(OUT_DIR / "11_hr_retrieval_memo.md",
                        header="Prototype evaluation on held-out questions; all tuning on a separate dev set.",
                        justification=f"'{winner}' passes both gates and has the best weighted balance of first-page "
                                      f"accuracy, ranking, refusals and index size ({', '.join(failed.index) or 'no candidate'} "
                                      f"failed a gate). Start the RAG pilot with it, and swap in a pre-trained embedding "
                                      f"model only if it beats this one on the same test set.",
                        extra_sections=extra)
card.to_json(OUT_DIR / "11_hr_retrieval_scorecard.json")
show("Memo written", memo.name)
print("\n" + "\n".join(memo.read_text(encoding="utf-8").splitlines()[:14]))
note("""The memo is the deliverable, not the code: requirements, a candidate table on
held-out data, the recommendation with its reason, the error analysis, and a
concrete plan for the RAG version with its risks. Notice how little of the
RAG plan is about the LLM: ingestion, chunking, metadata, retrieval,
refusal and evaluation are where RAG projects succeed or fail, and you have
now built and measured every one of them.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    tune.plot(ax=axes[0], marker="o")
    axes[0].set(xlabel="alpha (weight on embeddings)", ylabel="MRR on dev", title="Tuning the hybrid on dev")
    r = res[["recall@1", "recall@3", "oos_refused"]].astype(float)
    r.plot.bar(ax=axes[1])
    axes[1].axhline(0.90, color="r", ls="--", lw=1, label="recall@3 gate")
    axes[1].set(ylim=(0, 1.05), title="Candidates on TEST")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8)
    save_fig(fig, "11_hr_retrieval_scorecard")

banner("KEY TAKEAWAYS")
print("""  1. Split labelled questions into dev (all tuning) and test (reported once), plus out-of-scope questions.
  2. Compare retrievers with hard gates and weighted business criteria, not one metric.
  3. Error analysis (which pages get confused) tells you which fix to build next.
  4. A RAG plan is mostly retrieval engineering: ingestion, chunking, metadata, refusal, evaluation, monitoring.""")
