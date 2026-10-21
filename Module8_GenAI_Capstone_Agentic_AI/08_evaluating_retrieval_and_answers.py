"""
LAB 08 - Evaluating Retrieval and Generated Answers
===================================================

YOU CAN'T IMPROVE WHAT YOU DON'T MEASURE
    A RAG system can fail in two places, and the fixes are completely different:
        RETRIEVAL failed    the right article never reached the prompt    -> fix chunking, search, the index
        GENERATION failed   it was there, and the answer is still wrong   -> fix the prompt, ordering, the model
    So we measure them SEPARATELY, on questions whose right answer we know.

THE METRICS
    retrieval   recall@k (right article in the top k?), MRR (how high?)
    answer      fact recall    does the answer contain the facts a correct answer must contain?
                faithfulness   is every sentence of the answer supported by the retrieved context?
                citations      do the cited article ids include a right one?
                abstention     how often does it say "I don't know"?
    judge       an LLM grading answers 1-5: cheap at scale, and itself something to evaluate

    Faithfulness and correctness are DIFFERENT questions: an answer can be perfectly faithful to a stale
    article and still wrong.

LEVELS
    EASY   : One RAG configuration, every metric, and an error analysis: retrieval or generation?
    MEDIUM : Five configurations side by side. Which metric would have fooled you?
    HARD   : LLM-as-judge: does it agree with the answer key? Plot twist: what it really rewards.
"""
# %%
import json

import numpy as np
import pandas as pd

from genai_toolkit import (HybridEmbedder, PPMIEmbeddings, TfidfEmbedder, VectorIndex, cited_ids, default_retriever,
                           fact_recall, faithfulness, get_llm, kb_chunks, retrieval_report)
from ml_lab_utils import HAS_MPL, banner, make_kb, make_support_corpus, make_tickets, note, plt, save_fig, show, step

kb = make_kb()
tickets = make_tickets()
knowledge = tickets[tickets.intent.isin(["policy_question", "troubleshooting", "account"])]
dev, test = knowledge[knowledge.split == "dev"], knowledge[knowledge.split == "test"].reset_index(drop=True)
llm = get_llm("large")
SYSTEM = ("You are the support assistant for Lumora Home. Answer the customer's question using only the documents. "
          "Cite the document id in square brackets. If the documents do not contain the answer, say you don't know.")


def rag(question, index, k=3, system=SYSTEM):
    docs = []
    for d in (index.search(question, k=4 * k) if k else []):       # keep the top k DISTINCT articles
        if d["doc_id"] not in {x["doc_id"] for x in docs}:
            docs.append(d)
    docs = docs[:k]
    ctx = "\n".join(f'<document id="{d["doc_id"]}" date="{d["updated"]}">{d.get("raw", d["text"])}</document>' for d in docs)
    r = llm.create(system=system, messages=[{"role": "user", "content": f"{ctx}\n\n<question>{question}</question>"}])
    return r.text, docs, r.cost_usd


def evaluate(index, k=3, system=SYSTEM, df=test):
    rows = []
    for q in df.itertuples():
        a, docs, cost = rag(q.text, index, k, system)
        ctx = " ".join(d.get("raw", d["text"]) for d in docs)
        ids = list(dict.fromkeys(d["doc_id"] for d in docs))
        rows.append({"retrieved": ids, "gold": q.kb_ids, "answer": a, "fact recall": fact_recall(a, q.answer_facts),
                     "faithfulness": faithfulness(a, ctx) if docs else 0.0,
                     "citation hit": bool(set(cited_ids(a)) & set(q.kb_ids)),
                     "abstained": "don't know" in a.lower(), "cost": cost})
    out = pd.DataFrame(rows, index=df.index)
    out["retrieval hit"] = [bool(set(r) & set(g)) for r, g in zip(out.retrieved, out.gold)]
    return out


def summary(res):
    rr = retrieval_report(list(res.retrieved), list(res.gold), ks=(1, 3)) if res.retrieved.map(len).sum() else \
        {"recall@1": 0.0, "recall@3": 0.0, "MRR": 0.0}
    return {"recall@3": rr["recall@3"], "MRR": rr["MRR"], "fact recall": res["fact recall"].mean(),
            "faithfulness": res["faithfulness"].mean(), "citation hit": res["citation hit"].mean(),
            "abstained": res["abstained"].mean(), "$ per 1,000": res["cost"].sum() / len(res) * 1000}


# %% =========================================================================
# LEVEL 1 (EASY): Every metric for one configuration, and where it fails
# ============================================================================
banner("One RAG configuration, every metric", "EASY")
base = evaluate(default_retriever())
show("Test questions", len(test))
print(pd.Series(summary(base)).round(3).to_string())


def failure(r):
    if r["fact recall"] == 1:
        return "correct"
    if r["abstained"]:
        return "said 'I don't know'"
    if not r["retrieval hit"]:
        return "RETRIEVAL: right article not retrieved"
    return "GENERATION: retrieved, answered wrong"


base["outcome"] = base.apply(failure, axis=1)
outcomes = base.outcome.value_counts()
print(outcomes.to_string())
for kind in ["RETRIEVAL: right article not retrieved", "GENERATION: retrieved, answered wrong"]:
    ex = base[base.outcome == kind]
    if len(ex):
        r = ex.iloc[0]
        print(f"\n  {kind}\n    Q: {test.text[r.name]}\n    needs: {test.answer_facts[r.name]}   gold: {r.gold}   "
              f"retrieved: {r.retrieved}\n    A: {r.answer[:150]}")
n_ret, n_gen = outcomes.get("RETRIEVAL: right article not retrieved", 0), outcomes.get("GENERATION: retrieved, answered wrong", 0)
note(f"""Of {len(test)} test questions, {outcomes.get('correct', 0)} are answered correctly. The {n_ret + n_gen} failures split into
{n_ret} RETRIEVAL failures (the right article never reached the model: no prompt
can fix that) and {n_gen} GENERATION failures (the right article was there; the
model picked another sentence, a stale article, or an injected one). This
split tells you where to spend next week. One number ('{base['fact recall'].mean():.0%} correct') wouldn't.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Five configurations, side by side
# ============================================================================
banner("Five configurations, every metric", "MEDIUM")
items = kb_chunks(kb, max_words=None)
semantic = PPMIEmbeddings(dim=60).fit(make_support_corpus())
extra = [{**next(i for i in items if i["doc_id"] == d), "text": r.text} for r in dev.itertuples() for d in r.kb_ids]
kw_plus = TfidfEmbedder().fit([i["text"] for i in items + extra] + make_support_corpus(n=500))
expanded = VectorIndex(HybridEmbedder(kw_plus, semantic, 0.5)).add(
    [i for i in items + extra if i["status"] != "stale"])            # Lab 06's past questions, minus stale articles
configs = {
    "A closed-book (no retrieval)": (default_retriever(), 0, "You are the support assistant for Lumora Home."),
    "B RAG, k = 1": (default_retriever(), 1, SYSTEM),
    "C RAG, k = 3": (default_retriever(), 3, SYSTEM),
    "D RAG, k = 3, stale articles not indexed": (default_retriever(current_only=True), 3, SYSTEM),
    "E D + past questions in the index (Lab 06)": (expanded, 3, SYSTEM),
}
results = {name: (base if name.startswith("C") else evaluate(idx, k, s)) for name, (idx, k, s) in configs.items()}
board = pd.DataFrame({name: summary(r) for name, r in results.items()}).T
print(board.round(3).to_string())
stale_faithful = results["C RAG, k = 3"]
wrong_c = stale_faithful[(stale_faithful["fact recall"] < 1) & ~stale_faithful.abstained]
sf = wrong_c[wrong_c["faithfulness"] == 1]
note(f"""Read the table column by column. Retrieval metrics and fact recall move
together: better search (E) gives better answers ({board.iloc[4, 2]:.0%} vs {board.iloc[2, 2]:.0%}).
Faithfulness is the metric that would have fooled you: config C scores {board.iloc[2, 3]:.0%},
and {'every one' if len(sf) == len(wrong_c) else f'{len(sf)}'} of its {len(wrong_c)} wrong answers is 100% faithful: faithful to the stale
2024 policy, to the wrong troubleshooting article, or to the attacker's
community post. Faithfulness checks 'did it stick to the context?', not 'was
the context right?'. You need both, plus questions with known answers.
Closed-book (A) shows the floor.""")
if len(sf):
    r = sf.iloc[0]
    show("Faithful but wrong, e.g.", f"{test.text[r.name]!r} -> {r.answer[:110]!r}")

# %% =========================================================================
# LEVEL 3 (HARD): LLM-as-judge
# ============================================================================
banner("LLM-as-judge: who grades the grader?", "HARD")


def judge(question, answer_text, reference=None):
    ref = f"<reference>{'; '.join(reference)}</reference>\n" if reference else ""
    rubric = "" if reference else "Rubric: 5 = complete, correct and helpful; 1 = wrong or unhelpful.\n"
    user = (f"Grade the support answer from 1 to 5. {rubric}Reply in JSON: {{\"score\": n}}\n"
            f"<question>{question}</question>\n{ref}<answer>{answer_text}</answer>")
    r = llm.create(messages=[{"role": "user", "content": user}], max_tokens=20)
    return json.loads(r.text)["score"], r.cost_usd


res = results["E D + past questions in the index (Lab 06)"].copy()
res["judge, no reference"] = [judge(test.text[i], a)[0] for i, a in zip(res.index, res.answer)]
res["judge, with reference facts"] = [judge(test.text[i], a, test.answer_facts[i])[0] for i, a in zip(res.index, res.answer)]
res["correct"] = res["fact recall"] == 1
agree = res.groupby("correct")[["judge, no reference", "judge, with reference facts"]].mean()
agree.index = agree.index.map({True: "answer correct (key)", False: "answer wrong (key)"})
print(agree.round(2).to_string())
corr = res[["fact recall", "judge, no reference", "judge, with reference facts"]].corr(method="spearman")["fact recall"]
show("Rank correlation with the answer key: no reference / with reference",
     f"{corr['judge, no reference']:.2f} / {corr['judge, with reference facts']:.2f}")
note(f"""Without the right answer to compare against, the judge can't tell correct from
wrong: average score {agree.iloc[1, 0]:.1f} for correct answers, {agree.iloc[0, 0]:.1f} for wrong ones. Give it the
reference facts and it separates them ({agree.iloc[1, 1]:.1f} vs {agree.iloc[0, 1]:.1f}; rank correlation with the
key {corr['judge, with reference facts']:.2f}). A judge is a model: validate it against labelled examples before
trusting it with the other 10,000.""")

step("3.1  Plot twist: an A/B test the judge gets wrong")
FILLER = (" Thank you so much for reaching out to Lumora Home, we really appreciate your patience and your loyalty. "
          "We know how important your smart home is to you, and our whole team is always here to help with anything "
          "else you might need, any time of day.")
scored = res[~res.abstained]
ab = pd.DataFrame({"system A: answers as written": [judge(test.text[i], a, test.answer_facts[i])[0] for i, a in zip(scored.index, scored.answer)],
                   "system B: same answers + pleasantries": [judge(test.text[i], a + FILLER, test.answer_facts[i])[0]
                                                            for i, a in zip(scored.index, scored.answer)]}, index=scored.index)
summary_ab = pd.DataFrame({"fact recall (answer key)": [scored["fact recall"].mean(), np.mean([fact_recall(a + FILLER, test.answer_facts[i]) for i, a in zip(scored.index, scored.answer)])],
                           "average judge score": ab.mean().values}, index=ab.columns)
print(summary_ab.round(3).to_string())
pad = ab.groupby(scored.correct.map({True: "correct answers", False: "wrong answers"})).mean()
print(pad.round(2).to_string())
better, worse = (ab.iloc[:, 1] > ab.iloc[:, 0]).mean(), (ab.iloc[:, 1] < ab.iloc[:, 0]).mean()
note(f"""System B is system A plus two sentences of pleasantries: identical facts, same
fact recall ({summary_ab.iloc[0, 0]:.0%}). The judge prefers B ({summary_ab.iloc[1, 1]:.2f} vs {summary_ab.iloc[0, 1]:.2f}): higher on {better:.0%} of the
questions, lower on {worse:.0%}, even with the reference facts in hand; the lift is
largest on correct answers ({pad.iloc[0, 0]:.2f} -> {pad.iloc[0, 1]:.2f}). LLM judges are known to prefer longer,
more polite answers (verbosity bias). Optimise against this judge and you'd
ship the wordier system and call it an improvement. Check what can be
checked deterministically (facts, ids, amounts), use the judge for what
can't (tone, clarity), and re-validate it whenever the prompt or model changes.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    outcomes.plot.barh(ax=axes[0], color=["C2" if o == "correct" else "C7" if "know" in o else "C3" for o in outcomes.index])
    axes[0].invert_yaxis()
    axes[0].set(title="Config C: where the answers fail", xlabel="questions")
    board[["recall@3", "fact recall", "faithfulness"]].plot.bar(ax=axes[1], rot=0)
    axes[1].set_xticks(range(len(board)), [n.split()[0] for n in board.index])
    axes[1].set(ylim=(0, 1.05), title="Configs A-E: retrieval, correctness, faithfulness")
    axes[1].legend(fontsize=8)
    pad.T.plot.bar(ax=axes[2], rot=0)
    axes[2].set_xticks(range(2), ["A: as written", "B: + pleasantries"])
    axes[2].set(ylim=(1, 5.6), ylabel="average judge score (1-5)", title="Same facts, the judge prefers B")
    axes[2].legend(fontsize=8, loc="upper center", ncol=2)
    save_fig(fig, "08_evaluating_retrieval_and_answers")

banner("KEY TAKEAWAYS")
print("""  1. Build a test set with known answers (facts, article ids) before tuning anything.
  2. Measure retrieval and generation separately: their failures need different fixes.
  3. Faithfulness is not correctness: an answer can be loyal to the wrong document.
  4. An LLM judge needs a reference and validation against labels; without them it can't spot wrong answers.
  5. Judges reward length and polish: check facts deterministically, judge only what can't be checked.""")
