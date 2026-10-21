"""
MODULE 6 ASSIGNMENT - STARTER: An IT Help-Centre Search Prototype, and its Path to RAG
=======================================================================================
Read ASSIGNMENT.md first. The worked example on a different knowledge base (the HR handbook) is
11_assignment_worked_example_hr_retrieval_to_rag.py.

PART A (data, dev/test split, harness, baseline) and PART C (scorecard, rubric, report) are written for
you. Your work is PART B: build at least three retrievers, tune one on dev, analyse its errors, try the
mini RAG pipeline, open the test set once, and write the report.

Run it as often as you like: Part C prints the automated rubric every time.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_lab_utils import DATA_DIR, OUT_DIR, banner, make_knowledge_base, make_text_corpus, note, show, step
from unsup_toolkit import (ModelScorecard, PPMIEmbeddings, Requirement, chunk_text, df_to_md, extractive_answer,
                           retrieval_report)

# %% =========================================================================
# PART A - Brief, data, split, harness (done for you)
# ============================================================================
banner("PART A - The IT help centre and its questions")
articles, questions = make_knowledge_base("it", seed=6)
corpus = make_text_corpus("it", n_sentences=6000, seed=6)          # unlabelled IT text, for training embeddings
DATA_DIR.mkdir(exist_ok=True)
articles.to_csv(DATA_DIR / "it_help_centre_articles.csv", index=False)
show("Help-centre articles (incl. look-alike pages) / labelled questions", f"{len(articles)} / {len(questions)}")

rng = np.random.default_rng(6)
questions["split"] = "test"
for _, grp in questions.groupby("gold_article"):
    questions.loc[rng.choice(grp.index, len(grp) // 2, replace=False), "split"] = "dev"
dev = questions[questions.split == "dev"].reset_index(drop=True)
_TEST = questions[questions.split == "test"].reset_index(drop=True)     # sealed: use open_test_set()
OOS_DEV = ["how many days of annual leave do I get", "when is pay day", "can I bring my dog to work",
           "what is the canteen menu today", "how do I claim travel expenses", "who won the football match"]
_OOS_TEST = ["how long is parental leave", "what is the share price today", "where is the nearest train station",
             "how do I book a meeting room for a party", "what is the pension contribution", "is the gym open on sunday"]
dev.to_csv(DATA_DIR / "it_help_centre_dev_questions.csv", index=False)
show("Dev questions (for ALL tuning) / sealed test questions", f"{len(dev)} / {len(_TEST)}")
show("Out-of-scope questions: dev / sealed test", f"{len(OOS_DEV)} / {len(_OOS_TEST)}")
DOCS = (articles.title + ". " + articles.text).tolist()
IDS = articles.article_id.to_numpy()

REQUIREMENTS = [
    Requirement("right article in top 3", "recall@3", 0.90, higher_is_better=True, hard=True,
                why="the help widget shows three suggestions"),
    Requirement("latency", "ms_per_query", 20, hard=True, why="results appear while the user types"),
    Requirement("right article first", "recall@1", higher_is_better=True, weight=0.40, why="the RAG answer cites the top article"),
    Requirement("ranking quality", "MRR", higher_is_better=True, weight=0.20, why="the right article near the top"),
    Requirement("refuses out-of-scope", "oos_refused", higher_is_better=True, weight=0.30,
                why="HR or personal questions must not get an IT answer"),
    Requirement("index size", "index_kb", weight=0.10, why="the help centre will grow to 20,000 articles"),
]
ORIGINAL = {(r.name, r.threshold, r.weight) for r in REQUIREMENTS}
_state = {"candidates": {}, "families": {}, "dev": {}, "test": {}, "test_opened": 0, "tuned": [], "errors_for": set(),
          "rag_calls": []}


def register(name: str, score_fn, family: str, index_bytes: int, how: str = ""):
    """Register a retriever.
    score_fn(list_of_questions) -> numpy array (n_questions x n_articles) of similarities, columns in IDS order.
    family: 'keyword', 'embedding' or 'hybrid'.   index_bytes: memory the stored document vectors take."""
    assert family in {"keyword", "embedding", "hybrid"}, "family must be keyword, embedding or hybrid"
    S = np.asarray(score_fn(list(dev.question)))
    assert S.shape == (len(dev), len(IDS)), f"score_fn must return shape {(len(dev), len(IDS))}, got {S.shape}"
    t0 = time.perf_counter()
    for q in dev.question[:40]:
        score_fn([q])
    ms = (time.perf_counter() - t0) / 40 * 1000
    rep = retrieval_report([list(IDS[np.argsort(-row)]) for row in S], list(dev.gold_article))
    threshold = float(np.quantile(S.max(axis=1), 0.08))            # answer ~92% of in-scope dev questions
    oos = np.asarray(score_fn(OOS_DEV)).max(axis=1)
    _state["candidates"][name], _state["families"][name] = (score_fn, threshold), family
    _state["dev"][name] = {**rep, "ms_per_query": ms, "oos_refused": float(np.mean(oos < threshold)),
                           "index_kb": index_bytes / 1024}
    print(f"  [dev] {name:<34} recall@1 {rep['recall@1']:.3f}  recall@3 {rep['recall@3']:.3f}  MRR {rep['MRR']:.3f}  "
          f"{ms:.2f} ms  out-of-scope refused {np.mean(oos < threshold):.0%}  ({how})")


def tune_hybrid(score_a, score_b, grid=np.round(np.linspace(0, 1, 11), 1)) -> float:
    """Tune alpha in  alpha * score_a + (1 - alpha) * score_b  by MRR on DEV. Returns the best alpha."""
    A, B = np.asarray(score_a(list(dev.question))), np.asarray(score_b(list(dev.question)))
    mrr = {float(a): retrieval_report([list(IDS[np.argsort(-row)]) for row in a * A + (1 - a) * B],
                                      list(dev.gold_article))["MRR"] for a in grid}
    print("  [tune] MRR on dev by alpha: " + "  ".join(f"{a:.1f}:{m:.3f}" for a, m in mrr.items()))
    _state["tuned"].append(mrr)
    return max(mrr, key=mrr.get)


def error_analysis(name: str, n: int = 10) -> pd.DataFrame:
    """Dev questions the candidate gets wrong at rank 1, and the most frequent confusions."""
    S = np.asarray(_state["candidates"][name][0](list(dev.question)))
    order = np.argsort(-S, axis=1)
    t = articles.set_index("article_id").title
    out = dev.assign(retrieved=IDS[order[:, 0]],
                     rank_of_gold=[int(np.where(IDS[o] == g)[0][0]) + 1 for o, g in zip(order, dev.gold_article)])
    out = out[out.retrieved != out.gold_article]
    print(out.assign(gold=out.gold_article.map(t), got=out.retrieved.map(t))[["question", "gold", "got", "rank_of_gold"]]
          .head(n).to_string(index=False))
    print((out.gold_article.map(t) + "  ->  " + out.retrieved.map(t)).value_counts().head(5).to_string())
    _state["errors_for"].add(name)
    return out


def rag_answer(name: str, question: str, k: int = 3, threshold: float | None = None) -> dict:
    """Mini RAG with the candidate as retriever: refuse below the threshold (default: the one set on dev to
    answer ~92% of in-scope dev questions; pass your own to explore the trade-off), else answer extractively
    from the top-k articles, with a citation."""
    score_fn, dev_threshold = _state["candidates"][name]
    threshold = dev_threshold if threshold is None else threshold
    s = np.asarray(score_fn([question]))[0]
    top = np.argsort(-s)[:k]
    if s[top[0]] < threshold:
        out = {"answer": "I don't know: please contact the service desk.", "source": None, "score": float(s[top[0]])}
    else:
        passages = [(IDS[i], c) for i in top for c in chunk_text(DOCS[i], max_words=40, overlap=10)]
        out = extractive_answer(question, passages, _embed_for_answers)
        out["score"] = float(s[top[0]])
    _state["rag_calls"].append((question, out["source"]))
    print(f"  Q: {question}\n     -> {out['answer']}  [source {out['source']}, top score {out['score']:.2f}]")
    return out


def open_test_set() -> pd.DataFrame:
    """Score every registered candidate on the sealed test questions. Call it ONCE, after choosing on dev."""
    _state["test_opened"] += 1
    for name, (score_fn, threshold) in _state["candidates"].items():
        S = np.asarray(score_fn(list(_TEST.question)))
        rep = retrieval_report([list(IDS[np.argsort(-row)]) for row in S], list(_TEST.gold_article))
        oos = np.asarray(score_fn(_OOS_TEST)).max(axis=1)
        _state["test"][name] = {**rep, "ms_per_query": _state["dev"][name]["ms_per_query"],
                                "oos_refused": float(np.mean(oos < threshold)), "index_kb": _state["dev"][name]["index_kb"]}
    table = pd.DataFrame(_state["test"]).T
    print(table.round(3).to_string())
    return table


# a fixed sentence embedder for the extractive answer step (the same for every candidate, so it's a fair comparison)
_ans_emb = PPMIEmbeddings(dim=40).fit(corpus)
_embed_for_answers = lambda texts: _ans_emb.embed(list(texts))  # noqa: E731

step("A1  Baseline: does the question share words with the article title?")


def title_overlap(qs):
    title_words = [set(t.lower().split()) for t in articles.title]
    return np.array([[len(set(q.lower().split()) & tw) / (len(tw) or 1) for tw in title_words] for q in qs])


register("Baseline: title word overlap", title_overlap, "keyword", index_bytes=sum(len(t) for t in articles.title),
         how="share of title words in the question")

# %% =========================================================================
# PART B - YOUR WORK
# ============================================================================
banner("PART B - Retrievers, tuning, errors, RAG, test, report (your work)")

step("B1  At least THREE retrievers: at least one keyword AND one embedding-based")
# TODO: fit your vectorisers / embeddings on `corpus` and/or DOCS (never on dev or test questions), then register.
# Example of the expected style (uncomment, then build your own):
# tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(corpus + DOCS)
# D_tfidf = tfidf.transform(DOCS)
# tfidf_scores = lambda qs: (tfidf.transform(qs) @ D_tfidf.T).toarray()
# register("word TF-IDF", tfidf_scores, "keyword", index_bytes=D_tfidf.data.nbytes, how="unigrams, sublinear tf")
#
# emb = PPMIEmbeddings(dim=60).fit(corpus)
# D_emb = emb.embed(DOCS)
# emb_scores = lambda qs: emb.embed(qs) @ D_emb.T
# register("PPMI embeddings", emb_scores, "embedding", index_bytes=D_emb.nbytes, how="60-d PPMI + SVD")

step("B2  Tune a hybrid on DEV")
# TODO: alpha = tune_hybrid(emb_scores, tfidf_scores)
#       register("hybrid", lambda qs: alpha * emb_scores(qs) + (1 - alpha) * tfidf_scores(qs), "hybrid", ...)

step("B3  Error analysis of your best dev candidate")
# TODO: error_analysis("<name>")   - for the dev recommendation printed in Part C. Which articles get confused,
#       and why? (look at the look-alike 'X' pages)

step("B4  Try the mini RAG pipeline")
# TODO: at least three rag_answer("<name>", question) calls, including at least one out-of-scope question
#       that should be refused, e.g. rag_answer("hybrid", "how many days of annual leave do I get").
#       Not refused? Look at its top score and try rag_answer(..., threshold=<value>): what does a higher
#       threshold cost on in-scope questions? (Discuss it in RISKS.)

step("B5  Open the test set ONCE")
# TODO: test_table = open_test_set()

step("B6  The report")
# TODO: write each section in your own words, citing your numbers. Word minimums: RETRIEVAL RESULTS 60,
#       ERROR ANALYSIS 40, RAG DESIGN 120 (must cover chunking, metadata and evaluation), RISKS 40,
#       RECOMMENDATION 120 (must name the recommended retriever and cite at least 3 numbers).
REPORT = {
    "RETRIEVAL RESULTS": """
""",
    "ERROR ANALYSIS": """
""",
    "RAG DESIGN": """
""",
    "RISKS": """
""",
    "RECOMMENDATION": """
""",
}

# %% =========================================================================
# PART C - Scorecard, rubric and the report (done for you)
# ============================================================================
banner("PART C - Decide, self-check, write the report")
source = _state["test"] if _state["test_opened"] else _state["dev"]
card = ModelScorecard("IT help-centre search: which retriever should the prototype and the RAG assistant use?",
                      REQUIREMENTS)
for name, res in source.items():
    card.add(name, res)
board = card.evaluate()
show("Scorecard computed on", "TEST (sealed set)" if _state["test_opened"] else "DEV (open the test set in B5)")
print(board.drop(columns=["failed gates"]).round(3).to_string())
winner = card.recommend()
show("Recommended by the scorecard", winner)
dev_card = ModelScorecard(card.problem, REQUIREMENTS)
for name, res in _state["dev"].items():
    dev_card.add(name, res)
dev_winner = dev_card.recommend()                                  # what you would have chosen before the test set
show("Recommended on DEV (analyse this one's errors in B3)", dev_winner)

candidates = [n for n in _state["candidates"] if not n.startswith("Baseline")]
fams = {_state["families"][n] for n in candidates}
words = {k: v.split() for k, v in REPORT.items()}
rag_text = REPORT["RAG DESIGN"].lower()
rec = REPORT["RECOMMENDATION"]
checks = {
    ">= 3 retrievers besides the baseline (5 pts)": len(candidates) >= 3,
    "Keyword AND embedding-based retrievers (5 pts)": "keyword" in fams and bool(fams & {"embedding", "hybrid"}),
    "A hybrid tuned on dev and registered (5 pts)": bool(_state["tuned"]) and "hybrid" in fams,
    "Requirements not loosened (4 pts)": ORIGINAL <= {(r.name, r.threshold, r.weight) for r in REQUIREMENTS},
    "Test set opened exactly once (4 pts)": _state["test_opened"] == 1,
    "Error analysis of the recommended retriever (5 pts)": bool({winner, dev_winner} & _state["errors_for"]),
    ">= 3 RAG answers incl. >= 1 refusal (6 pts)": len(_state["rag_calls"]) >= 3
                                                   and any(src is None for _, src in _state["rag_calls"]),
    "Report: word minimums, RAG design covers chunking/metadata/evaluation, recommendation names the winner "
    "with >= 3 numbers (6 pts)":
        len(words["RETRIEVAL RESULTS"]) >= 60 and len(words["ERROR ANALYSIS"]) >= 40 and len(words["RISKS"]) >= 40
        and len(words["RAG DESIGN"]) >= 120 and all(t in rag_text for t in ["chunk", "metadata", "evaluat"])
        and len(words["RECOMMENDATION"]) >= 120 and winner is not None and winner in rec
        and sum(any(ch.isdigit() for ch in w) for w in words["RECOMMENDATION"]) >= 3,
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
points = sum(int(k.split("(")[-1].split()[0]) for k, v in checks.items() if v)
show("Automated rubric points", f"{points} / 40")

if all(checks.values()):
    sections = "\n".join([f"## {k.title()}\n\n{REPORT[k].strip()}\n" for k in ["RETRIEVAL RESULTS", "ERROR ANALYSIS",
                                                                              "RAG DESIGN", "RISKS"]]
                         + ["## Dev results (used for all tuning)\n", df_to_md(pd.DataFrame(_state["dev"]).T.round(3)), ""])
    card.to_markdown(OUT_DIR / "assignment_it_search_report.md", justification=rec.strip(), extra_sections=sections,
                     header=f"{len(articles)} articles; {len(dev)} dev questions for tuning, {len(_TEST)} sealed test "
                            f"questions (scored once); {len(_OOS_TEST)} out-of-scope test questions.")
    card.to_json(OUT_DIR / "assignment_it_search_report.json")
    print("  [saved] outputs/assignment_it_search_report.md + outputs/assignment_it_search_report.json")
else:
    note("""Not all checks pass yet - keep going with PART B. The report is written
automatically once every check is PASS.""")
