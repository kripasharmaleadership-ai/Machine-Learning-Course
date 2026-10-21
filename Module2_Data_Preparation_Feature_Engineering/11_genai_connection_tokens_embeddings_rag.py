"""
LAB 11 - From Feature Engineering to GenAI: Tokens, Vectors, Embeddings, RAG Context
=====================================================================================

THE BIG IDEA
    Models only consume NUMBERS. Classical ML turns rows into feature vectors;
    GenAI turns TEXT into token IDs and embedding vectors. Every lesson of this
    module carries over - the objects just changed:

    CLASSICAL ML                          ->  GENAI / LLM SYSTEMS
    ------------------------------------      --------------------------------------------
    categorical encoding (one-hot)        ->  TOKENIZATION (text -> token IDs)
    handle_unknown / rare categories      ->  out-of-vocabulary words -> sub-word tokens
    bag of features, TF-IDF weighting     ->  sparse VECTOR representations
    scaling / Normalizer                  ->  L2-normalised vectors, cosine similarity
    feature engineering / PCA             ->  EMBEDDINGS (learned dense features)
    curse of dimensionality               ->  vector-search quality, embedding size
    joining extra columns                 ->  METADATA ENRICHMENT (date, product, version)
    missing values / duplicates / noise   ->  empty docs, copy-pasted pages, boilerplate
    sensitive columns                     ->  PII redaction before indexing
    preprocessing pipeline                ->  INGESTION pipeline -> chunks -> index
    feature selection                     ->  CONTEXT SELECTION (retrieval, filters, budget)
    point-in-time correctness             ->  "use the CURRENT policy, not the 2024 one"
    evaluation on held-out data           ->  retrieval evaluation (hit@k) on a question set

LEVELS (fully offline - no API keys, no model downloads)
    EASY   : Tokenization - word vs character vs a BPE tokenizer trained from scratch.
    MEDIUM : Vector representations - bag-of-words, TF-IDF, cosine, dense "embeddings".
    HARD   : Context preparation for RAG - clean, redact, dedupe, enrich with metadata,
             chunk, retrieve, evaluate, and pack a cited prompt within a token budget.
"""
# %%
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from ml_lab_utils import (HAS_MPL, KB_FOOTER, RANDOM_STATE, banner, make_support_kb, note, plt, save_fig,
                          show, step)

docs, questions = make_support_kb()
corpus_text = " ".join(d["text"] for d in docs)

print(pd.DataFrame([
    ["one-hot encoding of a category", "token ID lookup (vocabulary)"],
    ["handle_unknown='ignore' / min_frequency", "sub-word tokens: no word is ever 'unknown'"],
    ["TF-IDF / scaling", "sparse text vectors, term weighting"],
    ["Normalizer (row-wise L2)", "cosine similarity for vector search"],
    ["engineered dense features / PCA", "embeddings"],
    ["adding columns from other tables", "metadata enrichment"],
    ["dedupe, impute, remove noise", "dedupe pages, drop empty docs, strip boilerplate"],
    ["feature selection", "context selection (top-k, filters, token budget)"],
], columns=["classical ML", "GenAI equivalent"]).to_string(index=False))

# %% =========================================================================
# LEVEL 1 (EASY): Tokenization
# ============================================================================
banner("Tokenization: how text becomes integer IDs", "EASY")

step("1.1  Word-level vocabulary = one-hot encoding of words")
words = re.findall(r"[a-z0-9]+", corpus_text.lower())
vocab = {w for w, c in Counter(words).items()}
show("Words in corpus / vocabulary size", f"{len(words)} / {len(vocab)}")
for test_word in ["refund", "refunds", "refundable", "esim", "5g", "reactivation", "refnd"]:
    print(f"  {test_word:<13} -> {'known id ' + str(sorted(vocab).index(test_word)) if test_word in vocab else '[UNK]  (out of vocabulary)'}")
note("""Word vocabularies hit the same wall as one-hot encoding: every new word,
plural, product name or typo is UNKNOWN in production. Character vocabularies
never have unknowns, but make sequences ~5x longer (cost and context!).
Modern LLMs use SUB-WORD tokens: frequent words stay whole, rare words split
into known pieces.""")

step("1.2  Train a Byte-Pair-Encoding (BPE) tokenizer from scratch")


def train_bpe(text: str, n_merges: int):
    """Classic BPE (Sennrich et al., 2016): repeatedly merge the most frequent adjacent pair."""
    word_freq = Counter(re.findall(r"[a-z0-9]+", text.lower()))
    splits = {w: list(w) + ["</w>"] for w in word_freq}
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, f in word_freq.items():
            s = splits[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        for w in splits:
            s, out, i = splits[w], [], 0
            while i < len(s):
                if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                    out.append(s[i] + s[i + 1])
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            splits[w] = out
    return merges


def bpe_tokenize(text: str, merges) -> list[str]:
    tokens = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        s = list(w) + ["</w>"]
        for a, b in merges:
            i, out = 0, []
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            s = out
        tokens += s
    return tokens


merges = train_bpe(corpus_text, 300)
print("  First 12 merges learned:", [a + b for a, b in merges[:12]])
for w in ["refund", "refundable", "reactivation", "esim", "refnd", "northstar"]:
    print(f"  {w:<13} -> {bpe_tokenize(w, merges)}")

step("1.3  The vocabulary-size trade-off (like choosing min_frequency)")
sample = " ".join(d["text"] for d in docs[:6])
rows = []
for n in [0, 50, 150, 300, 600]:
    m = train_bpe(corpus_text, n)
    rows.append({"merges": n, "vocabulary size": len(set(bpe_tokenize(corpus_text, m))),
                 "tokens for 6 docs": len(bpe_tokenize(sample, m)),
                 "chars per token": round(len(sample.replace(" ", "")) / len(bpe_tokenize(sample, m)), 2)})
print(pd.DataFrame(rows).to_string(index=False))
note("""More merges = bigger vocabulary = fewer tokens per text. GPT-class models
use ~50k-200k tokens (~4 English characters each). Tokens are the unit of COST,
LATENCY and CONTEXT-WINDOW size - and tokenizers trained on English split other
languages, code and product names into many more pieces.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Vector representations -> embeddings
# ============================================================================
banner("From sparse bag-of-words to dense embeddings", "MEDIUM")
texts = [re.sub(r"<[^>]+>", " ", d["text"]).replace(KB_FOOTER.strip(), "") for d in docs if d["text"].strip()]
ids = [d["doc_id"] for d in docs if d["text"].strip()]

step("2.1  Bag-of-words = one-hot per token, summed per document")
bow = CountVectorizer(stop_words="english").fit(texts)
Xb = bow.transform(texts)
show("Matrix shape (docs x vocabulary)", Xb.shape)
show("Share of zeros (sparsity)", f"{1 - Xb.nnz / np.prod(Xb.shape):.1%}")

step("2.2  TF-IDF = weighting (rare words matter more) + L2 row normalisation")
tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(texts)
Xt = tfidf.transform(texts)
show("Row norms after TF-IDF (Normalizer built in)", np.round(np.sqrt(Xt.multiply(Xt).sum(axis=1)).A1[:4], 3))
note("""IDF down-weights words that appear everywhere (like 'NorthStar', 'app') -
the text version of 'drop near-constant features'. Rows are L2-normalised so
that a dot product = COSINE similarity: long and short documents compare
fairly. Remember Normalizer from Lab 05? Wrong tool for tabular data, exactly
the right tool here.""")


def search(query, vectorizer, X, k=3):
    sims = cosine_similarity(vectorizer.transform([query]), X).ravel()
    top = np.argsort(-sims)[:k]
    return [(ids[i], round(float(sims[i]), 3)) for i in top]


step("2.3  Typos break word vectors; character n-grams survive them")
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts)
Xc = char.transform(texts)
for q in ["refund for a phone", "refnd for my fone", "how do i activate esim", "activte e-sim"]:
    print(f"  {q!r:28} word: {search(q, tfidf, Xt, 1)[0]}   char n-gram: {search(q, char, Xc, 1)[0]}")
note("""A word-level score of 0.0 means NO shared word - the 'top' result is then
arbitrary. Character n-grams ('ref', 'efu', 'und') still overlap with 'refnd',
so they find a refund policy. Representation choice = robustness to noise.""")

step("2.4  Dense 'embeddings': compress TF-IDF with SVD (Latent Semantic Analysis)")
lsa = make_pipeline(TruncatedSVD(8, random_state=RANDOM_STATE), Normalizer())
E = lsa.fit_transform(Xt)
show("Sparse dims -> dense dims", f"{Xt.shape[1]} -> {E.shape[1]}")
sim = pd.DataFrame(cosine_similarity(E), index=ids, columns=ids).round(2)
print(sim.loc[["KB-101", "KB-102", "KB-601", "KB-602", "KB-201"], ["KB-101", "KB-102", "KB-601", "KB-602", "KB-201"]]
      .to_string())
note("""Each document is now a dense 8-number vector; similar documents point in
similar directions (the two refund policies; the billing FAQ and its copy).
Production systems use NEURAL embedding models (hundreds to thousands of dims)
trained on huge corpora, so 'money back' lands near 'refund' even with zero
shared words. Conceptually it is the same move as Lab 08: a learned,
lower-dimensional representation where distance means something.""")

# %% =========================================================================
# LEVEL 3 (HARD): Context preparation for RAG
# ============================================================================
banner("RAG ingestion: clean -> redact -> dedupe -> enrich -> chunk -> evaluate", "HARD")

step("3.1  Data quality report for the knowledge base")
kb = pd.DataFrame([{"doc_id": d["doc_id"], "title": d["title"], "source": d["source"],
                    "chars": len(d["text"].strip()), "has_html": bool(re.search(r"<[a-z]+", d["text"])),
                    "has_email": bool(re.search(r"[\w.+-]+@[\w-]+\.\w+", d["text"])),
                    "meta_keys": ",".join(d["meta"]) or "(none)"} for d in docs])
print(kb.to_string(index=False))
note("""Same audit as a table: an EMPTY document (missing value), raw HTML
(noise), personal data (sensitive column), missing/inconsistent metadata
('Devices' vs 'devices', no product, no date) and - not visible here - a
near-duplicate page and an outdated policy version.""")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def strip_html(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def find_boilerplate(texts, min_share=0.5):
    """Sentences that appear in more than half of all documents = boilerplate
    (the text version of a near-constant column)."""
    df_count = Counter(s for t in texts for s in set(sentences(t)))
    return {s for s, c in df_count.items() if c / len(texts) > min_share}


def redact_pii(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    return re.sub(r"\+?\d[\d\-\s()]{7,}\d", "[PHONE]", text)


def shingles(text, k=5):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


PRODUCT_KEYWORDS = {"refund": "devices", "router": "devices", "roaming": "mobile", "esim": "mobile",
                    "fiber": "fiber", "bill": "billing", "contract": "account", "personal data": "account"}


def ingest(raw_docs, clean=True):
    """The RAG version of a preprocessing pipeline. Returns cleaned, enriched documents."""
    if not clean:
        return [dict(d, text=d["text"], **d["meta"]) for d in raw_docs]
    texts = [strip_html(d["text"]) for d in raw_docs if d["text"].strip()]
    boiler = find_boilerplate(texts)
    out = []
    for d in raw_docs:
        if not d["text"].strip():                                   # "missing value": skip + log
            print(f"     skipped empty document {d['doc_id']}")
            continue
        text = redact_pii(" ".join(s for s in sentences(strip_html(d["text"])) if s not in boiler))
        dup_of = next((o["doc_id"] for o in out if jaccard(shingles(o["text"]), shingles(text)) > 0.8), None)
        if dup_of:                                                  # "duplicate record": drop + log
            print(f"     dropped {d['doc_id']} as near-duplicate of {dup_of}")
            continue
        meta = dict(d["meta"])
        product = (meta.get("product") or "").lower() or next(
            (p for kw, p in PRODUCT_KEYWORDS.items() if kw in text.lower()), "general")
        out.append({"doc_id": d["doc_id"], "title": d["title"], "source": d["source"], "text": text,
                    "product": product, "updated": pd.Timestamp(meta.get("updated", "1900-01-01"))})
    latest = {}
    for o in out:                                                   # version resolution, per title
        key = o["title"].lower()
        latest[key] = max(latest.get(key, pd.Timestamp.min), o["updated"])
    for o in out:
        o["is_latest"] = o["updated"] == latest[o["title"].lower()]
    return out


step("3.2  Run the ingestion pipeline")
print("  boilerplate sentences found:", [s[:50] + "..." for s in find_boilerplate(
    [strip_html(d["text"]) for d in docs if d["text"].strip()])])
clean_docs = ingest(docs)
print(pd.DataFrame([{k: v for k, v in o.items() if k != "text"} for o in clean_docs])
      .assign(updated=lambda x: x.updated.dt.date).to_string(index=False))
print("  KB-701 after redaction:", next(o["text"] for o in clean_docs if o["doc_id"] == "KB-701")[200:330], "...")
note("""Enrichment added a normalised PRODUCT (inferred where missing), a parsed
DATE and an IS_LATEST flag (version resolution: the 2024 refund policy is
superseded). Personal data is redacted BEFORE indexing - once it is in a
vector index, it can be retrieved and repeated by the LLM. Names ('Jane Doe')
need an NER model; regexes catch only structured PII.""")

step("3.3  Chunking: fixed-size vs sentence-aware with overlap + title header")


def fixed_chunks(d, size=180):
    t = d["text"]
    return [t[i:i + size] for i in range(0, len(t), size)]


def sentence_chunks(d, max_chars=320, overlap=1, header=False):
    sents, chunks, cur = sentences(d["text"]), [], []
    for s in sents:
        if cur and len(" ".join(cur + [s])) > max_chars:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else []
        cur.append(s)
    if cur:
        chunks.append(cur)
    prefix = f"{d['title']} ({d.get('product', '')}, updated {d['updated'].date()}): " if header else ""
    return [prefix + " ".join(c) for c in chunks]


example = next(o for o in clean_docs if o["doc_id"] == "KB-201")
print("  FIXED 180-char chunk #2   :", repr(fixed_chunks(example)[1]))
print("  SENTENCE chunk #2 (+title):", repr(sentence_chunks(example, header=True)[1]))
note("""Fixed-size chunks cut sentences (and answers) in half. Sentence-aware
chunks keep facts intact; OVERLAP repeats one sentence so a fact at a boundary
survives; a TITLE/METADATA HEADER puts the context ('Travel Pass', 'mobile')
into every chunk - the chunk is the 'row', the header is 'joining the
dimension table'.""")

step("3.4  Retrieval evaluation: does the right chunk reach the LLM?")
GOLD = {"KB-601": {"KB-601", "KB-602"}}          # the copy also contains the true answer


def build_index(doc_list, chunker):
    rows = [{"doc_id": d["doc_id"], "text": c, "is_latest": d.get("is_latest", True)}
            for d in doc_list for c in chunker(d) if c.strip()]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
    return pd.DataFrame(rows), vec, vec.fit_transform([r["text"] for r in rows])


def evaluate(doc_list, chunker, latest_only=False, k=3):
    idx, vec, X = build_index(doc_list, chunker)
    hits1 = hits3 = stale = 0
    ctx_tokens = []
    for q in questions:
        sims = cosine_similarity(vec.transform([q["q"]]), X).ravel()
        if latest_only:
            sims = np.where(idx.is_latest.to_numpy(), sims, -1)    # METADATA FILTER
        top = idx.iloc[np.argsort(-sims)[:k]]
        gold = GOLD.get(q["gold"], {q["gold"]})
        ok = [(r.doc_id in gold) and (q["answer"].lower() in r.text.lower()) for r in top.itertuples()]
        hits1 += ok[0]
        hits3 += any(ok)
        stale += (q["gold"] == "KB-102") and ("KB-101" in set(top.doc_id))
        ctx_tokens.append(len(bpe_tokenize(" ".join(top.text), merges)))
    n_refund = sum(q["gold"] == "KB-102" for q in questions)
    return {"chunks in index": len(idx), "answer hit@1": hits1 / len(questions),
            "answer hit@3": hits3 / len(questions), "stale policy in top-3": f"{stale}/{n_refund}",
            "avg context tokens": int(np.mean(ctx_tokens))}


raw_docs = ingest(docs, clean=False)
configs = {
    "A. raw docs, fixed 180-char chunks": evaluate(raw_docs, fixed_chunks),
    "B. cleaned, fixed 180-char chunks": evaluate(clean_docs, fixed_chunks),
    "C. cleaned, sentence chunks + overlap": evaluate(clean_docs, sentence_chunks),
    "D. C + title header + is_latest filter": evaluate(
        clean_docs, lambda d: sentence_chunks(d, header=True), latest_only=True),
}
results = pd.DataFrame(configs).T
print(results.to_string())
note("""Read it like the ablation table in Lab 10:
 - Cleaning shrinks the index by ~a quarter (boilerplate, HTML, the duplicate page)
   at no loss - less to store, search and pay for, and no copied footer in answers.
 - Sentence chunks keep answers whole -> better hit@1.
 - The metadata filter removes the STALE 2024 refund policy. Without it, the LLM
   gets both '14 days' and '30 days' in its context for every refund question -
   and may well answer with the wrong one. No retrieval metric above shows that
   risk; only a check designed around the DATA (versions) does.
 - Headers cost tokens: richer chunks = more context per question. A trade-off.
An LLM can only be as good as the context you hand it.""", keep_lines=True)

step("3.5  Pack the final prompt: top-k chunks, citations, token budget")
TOKEN_BUDGET = 220
q = questions[0]["q"]
idx, vec, X = build_index(clean_docs, lambda d: sentence_chunks(d, header=True))
sims = np.where(idx.is_latest.to_numpy(), cosine_similarity(vec.transform([q]), X).ravel(), -1)
context, used = [], 0
for i in np.argsort(-sims)[:5]:
    piece = f"[{idx.doc_id.iloc[i]}] {idx.text.iloc[i]}"
    n_tok = len(bpe_tokenize(piece, merges))
    if used + n_tok > TOKEN_BUDGET:
        break
    context.append(piece)
    used += n_tok
prompt = ("You are a NorthStar support assistant. Answer ONLY from the context. "
          "Cite sources like [KB-123]. If the answer is not in the context, say you don't know.\n\n"
          "CONTEXT:\n" + "\n".join(context) + f"\n\nQUESTION: {q}\nANSWER:")
print(prompt)
show("Context tokens used / budget", f"{used} / {TOKEN_BUDGET}")
show("Source of the FIRST (top-ranked) chunk", context[0].split("]")[0] + "]")
note("""This string is what the LLM actually sees. Look at the ranking: the top
chunk may be about a DIFFERENT topic, matched only on shared words like 'ask
for'. Lexical TF-IDF has no idea what the question means - production systems
add neural embeddings and a re-ranker, and the prompt tells the model to cite
and to say 'I don't know'. Notice also that 'Jane Doe' survived redaction.
Every upstream decision - tokenizer, cleaning, PII, dedupe, metadata, chunking,
filtering, ranking, budget - shaped this string. 'Prompt engineering' is mostly
CONTEXT engineering, and context engineering is data preparation.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    r = results[["answer hit@1", "answer hit@3"]].astype(float)
    r.plot.barh(ax=ax, color=["#4c78a8", "#59a14f"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("RAG retrieval quality by ingestion choices")
    save_fig(fig, "11_rag_retrieval_eval")

banner("KEY TAKEAWAYS")
print("""  1. Tokenization is categorical encoding for text; sub-words solve 'unknown category'.
  2. Text vectors are features: TF-IDF weights, L2-normalise for cosine, embeddings compress.
  3. A RAG ingestion pipeline IS a preprocessing pipeline: clean, redact, dedupe, enrich, chunk.
  4. Metadata (date, version, product) enables filters that fix stale or wrong context.
  5. Evaluate retrieval with a question set (hit@k), exactly like a held-out test set.""")
