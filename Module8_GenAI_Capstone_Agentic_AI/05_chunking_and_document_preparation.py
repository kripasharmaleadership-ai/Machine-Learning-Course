"""
LAB 05 - Chunking and Document Preparation
==========================================

RETRIEVAL-AUGMENTED GENERATION (RAG), LABS 05-08
    Lab 01 showed the model doesn't know Lumora's policies. RAG fixes that at question time:
        1. PREPARE  the documents: clean them, split them into chunks, keep their metadata     <- this lab
        2. EMBED    every chunk as a vector; index the vectors                                  Lab 06
        3. RETRIEVE the chunks closest to the question                                          Lab 06
        4. GENERATE an answer from the retrieved chunks, with citations                         Lab 07
        5. EVALUATE retrieval and answers separately                                            Lab 08

WHAT WE GET FROM THE BUSINESS
    Not tidy articles: one long "Help Centre Handbook" exported from a document tool, with page footers in the
    middle of sentences, words hyphenated across line breaks, last year's policies still inside, and a community
    post nobody vetted. That's normal. Garbage in, garbage retrieved.

THE QUESTIONS
    166 tickets that the help centre should answer (policy, troubleshooting, account), each with the facts a
    correct answer must contain (e.g. "30 days"). We measure two things for every way of chunking:
        answer in top 3   is a retrieved chunk carrying the needed fact?           (retrieval)
        answer correct    does the model's answer from those chunks contain it?   (generation)

LEVELS
    EASY   : From a messy export to clean text. Why not just send the whole handbook?
    MEDIUM : Chunk size and overlap: the trade-off between finding the fact and keeping its context.
    HARD   : Structure-aware chunks with metadata, a date filter for stale policies, and a plot twist.
"""
# %%
import re

import numpy as np
import pandas as pd

from genai_toolkit import (HybridEmbedder, PPMIEmbeddings, TfidfEmbedder, VectorIndex, chunk_text, count_tokens,
                           fact_recall, get_llm)
from ml_lab_utils import HAS_MPL, banner, make_kb, make_support_corpus, make_tickets, note, plt, save_fig, show, step

kb = make_kb()
tickets = make_tickets()
questions = tickets[tickets.intent.isin(["policy_question", "troubleshooting", "account"])].reset_index(drop=True)
llm = get_llm("large")
SEMANTIC = PPMIEmbeddings(dim=60).fit(make_support_corpus())      # word vectors from support text (Lab 06)
BACKGROUND = make_support_corpus(n=500)


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower())


def build_index(chunks: list[dict]) -> VectorIndex:
    """A hybrid keyword + semantic index over the chunks (how it works: Lab 06)."""
    keyword = TfidfEmbedder().fit([c["text"] for c in chunks] + BACKGROUND)
    return VectorIndex(HybridEmbedder(keyword, SEMANTIC, 0.5)).add(chunks)


def evaluate(chunks: list[dict], k: int = 3, where=None) -> dict:
    """For each question: retrieve k chunks, check whether the needed fact is in them, generate an answer."""
    index = build_index(chunks)
    found, correct, tokens = [], [], []
    for q in questions.itertuples():
        top = index.search(q.text, k=k, where=where)
        context = " ".join(c["text"] for c in top)
        found.append(np.mean([norm(f) in norm(context) for f in q.answer_facts]))
        tokens.append(count_tokens(context))
        docs = "\n".join(f'<document id="{c.get("id", i)}" date="{c.get("updated", "")}">{c["text"]}</document>'
                         for i, c in enumerate(top))
        ans = llm.create(system="Answer the customer's question using only the documents.",
                         messages=[{"role": "user", "content": f"{docs}\n\n<question>{q.text}</question>"}]).text
        correct.append(fact_recall(ans, q.answer_facts))
    return {"chunks": len(chunks), "answer in top 3": float(np.mean(found)), "answer correct": float(np.mean(correct)),
            "context tokens per question": float(np.mean(tokens))}


# %% =========================================================================
# LEVEL 1 (EASY): From a messy export to clean text
# ============================================================================
banner("What the business hands you: a messy handbook export", "EASY")
clean_handbook = "\n\n".join(f"## {d.title}\nLast updated: {d.updated}\n{d.text}" for d in kb.itertuples())


def export_like_a_pdf(text: str, words_per_page: int = 90, seed: int = 0) -> str:
    """Simulate a PDF export: long words hyphenated across line breaks, a footer at the end of every page."""
    rng = np.random.default_rng(seed)
    out, n, page = [], 0, 1
    for line in text.split("\n"):
        cur = []
        for w in line.split(" "):
            n += 1
            if len(w) >= 8 and w.isalpha() and rng.random() < 0.3:  # a long word broken across two lines
                cut = len(w) // 2
                cur.append(w[:cut] + "-\n" + w[cut:])
            else:
                cur.append(w)
            if n % words_per_page == 0:
                cur.append(f"\nLumora Home Help Centre | Internal | Page {page} of 12\n")
                page += 1
        out.append(" ".join(cur))
    return "\n".join(out)


raw = export_like_a_pdf(clean_handbook)
show("Handbook: words / approximate tokens", f"{len(clean_handbook.split()):,} / {count_tokens(clean_handbook):,}")
show("Articles inside (incl. 2 stale, 2 look-alike, 1 community)", len(kb))
i = raw.find("Page 2")
print("  A piece of the raw export:\n    " + raw[i - 160:i + 120].replace("\n", "\n    "))
facts = sorted({f for fs in questions.answer_facts for f in fs})
broken = [f for f in facts if norm(f) not in norm(raw.replace("\n", " "))]
show("Answer facts the raw export no longer contains intact", f"{len(broken)} of {len(facts)}: {broken[:3]}")


def clean_export(text: str) -> str:
    text = re.sub(r" ?\nLumora Home Help Centre \| Internal \| Page \d+ of \d+\n ?", " ", text)   # footers
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                                                 # re-join hyphenation
    return re.sub(r"[ \t]+", " ", text).strip()


cleaned = clean_export(raw)
still_broken = [f for f in facts if norm(f) not in norm(cleaned.replace("\n", " "))]
show("After cleaning: facts still broken", len(still_broken))
show("Cleaned text matches the source (ignoring spacing)", norm(cleaned) == norm(clean_handbook))
note(f"""A footer dropped into the middle of a sentence, or 'busi-' + 'ness' across a line
break, silently destroys the exact phrase a question needs: {len(broken)} of our {len(facts)}
answer facts were no longer findable in the raw export (a keyword index can't
match 'busi-ness'; an embedding sees two unknown fragments). Two regular
expressions fix it here; in real projects, inspect a sample of every new
source before indexing it. Clean text is the cheapest accuracy you'll get.""")

step("1.1  Why not just put the whole handbook in every prompt?")
whole = evaluate([{"id": "HANDBOOK", "text": cleaned}], k=1)
show("Whole handbook as context: answer correct", f"{whole['answer correct']:.1%}")
show("Context tokens per question", f"{whole['context tokens per question']:,.0f}")
show("Cost per 10,000 questions, large model (input only)", f"${whole['context tokens per question'] * 10_000 * 4 / 1e6:,.2f}")
note(f"""The whole handbook (~{whole['context tokens per question']:,.0f} tokens) fits easily in a modern context window, and
the answer is ALWAYS somewhere in it. Yet answers are correct only {whole['answer correct']:.0%} of
the time: the model has to find one sentence among a thousand words, next to a
stale 2024 version of the same policy. And it pays for every one of those
tokens on every question. Help centres grow; chunking and retrieval keep the
prompt small and focused.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Chunk size and overlap
# ============================================================================
banner("Chunk size and overlap", "MEDIUM")


def fixed_windows(text: str, size: int, overlap: int) -> list[dict]:
    w = text.split()
    stride = max(1, size - overlap)
    return [{"id": f"W{i}", "text": " ".join(w[i:i + size])} for i in range(0, max(1, len(w) - overlap), stride)]


rows = {}
for size in [15, 30, 60, 120, 240]:
    for overlap in [0, size // 4]:
        rows[(size, overlap)] = evaluate(fixed_windows(cleaned, size, overlap))
sweep = pd.DataFrame(rows).T
sweep.index.names = ["words per chunk", "overlap"]
print(sweep.round(3).to_string())
best = sweep["answer correct"].idxmax()
note(f"""Tiny chunks (15 words) are easy to match but rarely contain the whole fact:
the answer is in the top 3 only {sweep.iloc[0, 1]:.0%} of the time. Big chunks (240 words) almost
always contain it ({sweep.iloc[8, 1]:.0%}), yet the answers are correct only {sweep.iloc[8, 2]:.0%} of the time: each
chunk mixes several articles, so the model gets the right fact AND the wrong
ones, and the context costs {sweep.iloc[8, 3] / sweep.iloc[4, 3]:.0f}x more than with 60-word chunks. The best fixed
setting here is {best[0]} words with {best[1]} words of overlap ({sweep.loc[best, 'answer correct']:.0%}). Overlap repeats
words across a boundary so a fact cut in two survives in one piece; here it
helps some sizes and hurts others: it's a setting to test, not a rule.""")

# %% =========================================================================
# LEVEL 3 (HARD): Structure-aware chunks, metadata, and a plot twist
# ============================================================================
banner("Chunk along the document's own structure, and keep its metadata", "HARD")


def sections_with_metadata(text: str) -> list[dict]:
    """Split at the headings; keep the title and the 'Last updated' date as metadata on every chunk."""
    out = []
    for k, sec in enumerate(re.split(r"\n(?=## )", text)):
        title = re.match(r"## (.*)", sec).group(1)
        updated = re.search(r"Last updated: (\S+)", sec).group(1)
        body = re.sub(r"^## .*\n|Last updated: \S+\n?", "", sec).strip()
        out.append({"id": f"S{k:02d}", "title": title, "updated": updated, "text": f"{title}. {body}"})
    return out


secs = sections_with_metadata(cleaned)
show("Sections", len(secs))
show("A section chunk", {k: (v[:70] + "...") if k == "text" else v for k, v in secs[0].items()})
sentence_packed = [{"id": f"P{i}", "text": c} for i, c in enumerate(chunk_text(cleaned.replace("\n", " "), 40, 10))]
rows = {f"fixed windows, {best[0]} words, overlap {best[1]}": sweep.loc[best].to_dict(),
        "sentence-packed, ~40 words": evaluate(sentence_packed),
        "one chunk per section (article)": evaluate(secs),
        "sections + filter: updated in the last 12 months": evaluate(secs, where=lambda c: c["updated"] >= "2025-03-16"),
        "whole handbook": whole}
struct = pd.DataFrame(rows).T
print(struct.round(3).to_string())
s, f = struct.iloc[2], struct.iloc[3]
note(f"""Splitting at the document's own headings gives chunks that each answer ONE
question, carry their title, and never mix two policies: answers correct
{s['answer correct']:.0%}, the best so far, with small contexts. Because each chunk keeps its
'Last updated' date as metadata, we can also FILTER: drop anything older than
12 months and the stale 2024 policies can no longer be retrieved ({f['answer correct']:.0%}).
Fixed windows can't do this: a window spanning three articles has no single date.""")

step("3.1  Plot twist: which section ranks FIRST for a shipping-time question?")
index = build_index(secs)
kind = questions.answer_facts.map(lambda fs: "return window ('30 days')" if "30 days" in fs else
                                  "shipping time ('3-5 business days')" if "3-5 business days" in fs else None)
sub = questions[kind.notna()]
first = pd.crosstab(pd.Series([index.search(t, k=1)[0]["title"] for t in sub.text], index=sub.index, name="top-ranked section"),
                    kind[kind.notna()].rename("question about"))
print(first.to_string())
ship, ret = first["shipping time ('3-5 business days')"], first["return window ('30 days')"]
n_cur = ship.get("Shipping times and costs", 0)
note(f"""Return-window questions find the current returns policy ({ret.get('Returns policy', 0)} of {ret.sum()}). Shipping-time
questions {'NEVER rank' if n_cur == 0 else 'rarely rank'} the current shipping article first ({n_cur} of {ship.sum()}): the 2024
version wins {ship.get('Shipping times and costs (2024)', 0)} times (it's shorter and says nothing else, so it's the closest
match to a short question) and 'Changing your delivery address' wins {ship.get('Changing your delivery address', 0)}
(customers say 'delivery', the policy says 'shipping': Lab 06). Similarity
measures how alike two texts are, never which one is true or current. So
freshness must be metadata you filter on (or a 'prefer the newest document'
rule, Lab 07), and the cleanest fix is upstream: archive superseded documents
instead of indexing them.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for overlap_on, ls in [(False, "o-"), (True, "s--")]:
        sub = sweep[(sweep.index.get_level_values(1) > 0) == overlap_on]
        sizes = sub.index.get_level_values(0)
        lab = "with 25% overlap" if overlap_on else "no overlap"
        axes[0].plot(sizes, sub["answer in top 3"], ls, color="C0", label=f"answer in top 3, {lab}")
        axes[0].plot(sizes, sub["answer correct"], ls, color="C3", label=f"answer correct, {lab}")
    axes[0].set(xscale="log", xlabel="words per chunk (log)", ylim=(0, 1.05), title="Chunk size: found vs answered")
    axes[0].set_xticks([15, 30, 60, 120, 240], ["15", "30", "60", "120", "240"])
    axes[0].minorticks_off()
    axes[0].legend(fontsize=7)
    axes[1].scatter(sweep["context tokens per question"], sweep["answer correct"], color="C0", label="fixed windows")
    axes[1].scatter(struct["context tokens per question"].iloc[1:], struct["answer correct"].iloc[1:], color="C3",
                    label="sentence / sections / whole")
    for name, r in struct.iloc[1:].iterrows():
        axes[1].annotate(name.split(",")[0].split(":")[0][:22], (r["context tokens per question"], r["answer correct"]),
                         fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[1].set(xscale="log", xlabel="context tokens per question (log)", ylabel="answer correct",
                title="Quality vs cost per question")
    axes[1].legend(fontsize=7, loc="lower right")
    short = ["fixed windows", "sentence-packed", "sections", "sections\n+ date filter", "whole handbook"]
    axes[2].barh(short, struct["answer correct"], color=["C0", "C0", "C2", "C2", "C7"])
    axes[2].invert_yaxis()
    axes[2].set(xlim=(0, 1), title="Structure-aware chunks and metadata")
    save_fig(fig, "05_chunking_and_document_preparation")

banner("KEY TAKEAWAYS")
print("""  1. Inspect and clean every source: footers and hyphenation silently break the facts you need.
  2. Stuffing everything into the prompt is expensive and still answers worse than focused context.
  3. Chunk size is a trade-off: small chunks lose the fact's context, big chunks mix topics and cost more.
  4. Overlap is a setting to test, not a rule.
  5. Chunk along the document's structure and keep metadata (title, date, status) on every chunk.
  6. Similarity is not truth or freshness: filter stale content, or better, don't index it.""")
