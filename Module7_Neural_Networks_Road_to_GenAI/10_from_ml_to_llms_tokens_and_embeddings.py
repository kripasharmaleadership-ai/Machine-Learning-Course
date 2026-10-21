"""
LAB 10 - From ML to LLMs: Tokenization and Embeddings
=====================================================

AN LLM IS A CLASSIFIER YOU ALREADY KNOW, APPLIED OVER AND OVER
    input        the text so far
    features     LEARNED: token embeddings, refined by transformer blocks (Labs 07-09)
    output       a probability for every token in the vocabulary (softmax over ~50,000 classes, Lab 03)
    training     cross-entropy on the token that actually came next (Lab 03), by backprop (Lab 04) and Adam (Lab 05)
    generation   pick a token, append it, classify again (Lab 13)

THE FIRST TWO STEPS
    1. TOKENIZATION: text -> a list of integer ids. Not words, not characters: SUBWORD pieces learned from data
       (byte-pair encoding, BPE). Frequent words are one token; rare words are spelled with several pieces.
    2. EMBEDDING: each id -> a row of a learned matrix (vocab x d). Equivalent to one-hot times a matrix. The
       vectors are trained with the rest of the network; words used alike end up close (Module 6, Lab 08).

OUR INVENTED WORLD (used in Labs 10-13 and 15)
    48 countries with a capital, a language and a main export, written as ~1,200 short sentences, plus 1,500
    "map: key value ..." lines for in-context learning. 10 more countries appear only in small talk, never
    with a fact. We know every fact, so we can check every answer.

LEVELS
    EASY   : Characters, words or BPE? Vocabulary size, sequence length, and unknown words.
    MEDIUM : Embeddings: a lookup table; what the trained TinyGPT's embeddings learned.
    HARD   : Tokenization quirks that leak into LLM behaviour: spaces, digits, spelling, cost, vocabulary size.
"""
# %%
import re
from collections import Counter

import numpy as np
import pandas as pd

from ml_lab_utils import HAS_MPL, LANGUAGES, EXPORTS, banner, note, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, BPETokenizer, tokenizer_training_text, world_corpus, world_tokenizer

text, sentences, facts = world_corpus()
tok = world_tokenizer()
lines = text.strip().split("\n")
train_lines, held_lines = lines[: int(0.9 * len(lines))], lines[int(0.9 * len(lines)):]

# %% =========================================================================
# LEVEL 1 (EASY): Three ways to cut text into tokens
# ============================================================================
banner("Characters, words or subwords?", "EASY")
example = "The capital of " + facts.country.iloc[0] + " is " + facts.value.iloc[0] + "."
show("Example sentence", example)
show("Characters", list(example)[:24] + ["..."])
show("Words", example.split())
show("BPE tokens (our 400-token vocabulary)", tok.tokens(example))
show("BPE ids", tok.encode(example))

step("1.1  Train on 90% of the lines, test on the other 10%")
char_vocab = set("\n".join(train_lines))
word_vocab = set(w for ln in train_lines for w in re.findall(r"\w+|[^\w\s]", ln))
held_words = [w for ln in held_lines for w in re.findall(r"\w+|[^\w\s]", ln)]
bpe = BPETokenizer().train("\n".join(train_lines), vocab_size=400)
held_text = "\n".join(held_lines)
rows = {
    "characters": {"vocabulary size": len(char_vocab), "tokens per held-out line": len(held_text) / len(held_lines),
                   "unknown tokens in held-out text": np.mean([c not in char_vocab for c in held_text])},
    "words": {"vocabulary size": len(word_vocab), "tokens per held-out line": len(held_words) / len(held_lines),
              "unknown tokens in held-out text": np.mean([w not in word_vocab for w in held_words])},
    "BPE (400 merges)": {"vocabulary size": len(bpe.vocab), "tokens per held-out line": len(bpe.encode(held_text)) / len(held_lines),
                         "unknown tokens in held-out text": np.mean(np.array(bpe.encode(held_text)) == 0)},
}
tt = pd.DataFrame(rows).T
print(tt.to_string(formatters={"vocabulary size": "{:,.0f}".format, "tokens per held-out line": "{:.1f}".format,
                               "unknown tokens in held-out text": "{:.2%}".format}))
note(f"""Characters: tiny vocabulary, nothing is ever unknown, but sequences are long
(every line is ~{tt.iloc[0, 1]:.0f} steps of attention). Words: short sequences, but any word not
seen in training (a new name, a typo) is UNKNOWN: rare here ({tt.iloc[1, 2]:.2%}) because our world
is tiny and repetitive, but constant in real text. BPE
sits between: a fixed {len(bpe.vocab)}-token vocabulary, sequences ~{tt.iloc[0, 1] / tt.iloc[2, 1]:.0f}x shorter than
characters, and nothing unknown, because a new word can always be spelled
from smaller pieces. GPT-2 uses 50,257 BPE tokens; newer LLMs 100,000-200,000.""")

step("1.2  How BPE builds its vocabulary: the first merges")
n_chars = len(set(tokenizer_training_text()))
BPETokenizer().train(tokenizer_training_text(), vocab_size=n_chars + 1 + 12, verbose=12)
note("""BPE starts from single characters and repeatedly merges the most frequent
adjacent pair into a new token, and merged tokens get merged again. After
enough merges, every common word is one token and rare words are a few
pieces. The vocabulary is LEARNED from the corpus: a tokenizer trained on
English splits other languages into more, shorter pieces (see Level 3).""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Embeddings
# ============================================================================
banner("Embeddings: one learned vector per token", "MEDIUM")
if not HAS_TORCH:
    raise SystemExit("Levels 2-3 need PyTorch:  pip install torch")
import torch

from nn_toolkit import world_lm

world = world_lm("base")
E = world.model.tok.weight.detach().numpy()
show("Embedding matrix (vocabulary x dimensions)", E.shape)
ids = tok.encode(" tea")
onehot = np.zeros(len(tok.vocab))
onehot[ids[0]] = 1
show("' tea' -> id", ids)
show("one-hot @ E  ==  E[id]  (an embedding is a lookup)", bool(np.allclose(onehot @ E, E[ids[0]])))
note("""The first layer of every LLM is a table with one row per token. Looking up a
row is the same as multiplying a one-hot vector by the table, so the table is
just a weight matrix, trained by backprop like every other. Nobody tells the
model what any token means.""")

step("2.1  What did TinyGPT's embeddings learn? Nearest neighbours by cosine")
single = lambda w: len(tok.encode(" " + w)) == 1
groups = {"export": [w for w in EXPORTS if single(w)], "language": [w for w in LANGUAGES if single(w)]}
En = E / np.linalg.norm(E, axis=1, keepdims=True)
word_ids = {w: tok.encode(" " + w)[0] for g in groups.values() for w in g}
for w in [groups["export"][0], groups["language"][0]]:
    sims = En @ En[word_ids[w]]
    nn_ = [tok.vocab[i] for i in np.argsort(-sims)[1:7]]
    show(f"Nearest tokens to ' {w}'", nn_)
rows = {}
cand = list(word_ids)
for g, ws in groups.items():
    hits = []
    for w in ws:
        sims = {o: float(En[word_ids[w]] @ En[word_ids[o]]) for o in cand if o != w}
        top = sorted(sims, key=sims.get, reverse=True)[:3]
        hits.append(np.mean([o in ws for o in top]))
    rows[g] = {"single-token words": len(ws), "top-3 neighbours from the same group": np.mean(hits),
               "chance": (len(ws) - 1) / (len(cand) - 1)}
emb_tab = pd.DataFrame(rows).T
print(emb_tab.round(3).to_string())
note(f"""Among all export and language words, an export's nearest neighbours are mostly
other exports ({emb_tab.iloc[0, 1]:.0%} vs {emb_tab.iloc[0, 2]:.0%} by chance), and languages cluster with languages. The
model was only ever trained to predict the next token, but words that appear
in the same slots ('known for its ___', 'people speak ___') get similar vectors,
because that makes prediction easier. This is Module 6's distributional
hypothesis, learned by gradient descent instead of PPMI + SVD.""")

# %% =========================================================================
# LEVEL 3 (HARD): Tokenization quirks that leak into LLM behaviour
# ============================================================================
banner("Tokenization quirks you will meet in real LLMs", "HARD")
country = facts.country.iloc[0]
step("3.1  A leading space makes a different token")
for s in [country, " " + country, country.lower(), country.upper()]:
    show(f"{s!r}", tok.tokens(s))
note("""' Name' (with its space, as in the middle of a sentence) and 'Name' at the start
of a line are different token sequences, and lower or upper case shatters the
word into pieces the model rarely saw. Real LLMs behave measurably worse on
odd casing and spacing for exactly this reason.""")

step("3.2  Digits, spelling, and why LLMs miscount letters")
show("'2026' ->", tok.tokens("2026"))
long_name = max(facts.country, key=len)
show(f"'{long_name}' ->", tok.tokens(" " + long_name))
show(f"How many letters in '{long_name}'?", f"{len(long_name)}, but the model sees {len(tok.encode(' ' + long_name))} tokens, not letters")
note("""Our pre-tokenizer splits numbers into single digits (as GPT-4-era tokenizers
do in groups of up to three). The model never sees letters inside a token,
only the token id, so 'how many r's in strawberry' is a question about
spelling the model has to have learned separately. Many famous LLM failures
are tokenization failures.""")

step("3.3  Tokens are the unit of cost: common vs rare words")
common = Counter(re.findall(r"[a-z]+", text)).most_common(30)
rows = {"30 most common lower-case words": np.mean([len(tok.encode(" " + w)) for w, _ in common]),
        "country names (invented)": np.mean([len(tok.encode(" " + c)) for c in facts.country.unique()]),
        "made-up words never seen ('blorptastic', 'quenzibar')": np.mean([len(tok.encode(" " + w)) for w in ["blorptastic", "quenzibar"]])}
print(pd.Series(rows, name="tokens per word").round(2).to_string())
note("""Frequent words cost one token; rare names cost several; unseen words cost the
most. Real tokenizers are trained mostly on English, so the same sentence in
Hindi, Tamil or Amharic can cost 2-10x more tokens than in English: more money
per API call, less text in the same context window, and often weaker answers.""")

step("3.4  The vocabulary-size trade-off")
rows = {}
for v in [100, 400, 1500]:
    b = BPETokenizer().train("\n".join(train_lines), vocab_size=v)
    n_tok = len(b.encode(held_text))
    rows[v] = {"tokens per held-out line": n_tok / len(held_lines), "words that fit in a 64-token context":
               64 / (n_tok / len(held_words)), "embedding parameters at d = 128": len(b.vocab) * 128}
vs = pd.DataFrame(rows).T
vs.index.name = "vocabulary size"
print(vs.to_string(formatters={"tokens per held-out line": "{:.1f}".format, "words that fit in a 64-token context": "{:.0f}".format,
                               "embedding parameters at d = 128": "{:,.0f}".format}))
note(f"""A bigger vocabulary makes sequences shorter (more text per context window,
cheaper attention), but costs a bigger embedding table, and each rare token
gets fewer training examples. Plot twist for small models: our world needs
only a few hundred tokens; at {vs.index[-1]:,} the vocabulary runs out of useful merges
(it stops when no pair repeats). Real LLMs pick 50k-200k as the balance for
trillions of tokens across many languages and code.""")

if HAS_MPL:
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ws = groups["export"] + groups["language"]
    P2 = PCA(2, random_state=0).fit_transform(E[[word_ids[w] for w in ws]])
    for g, c in [("export", "C1"), ("language", "C0")]:
        msk = np.array([w in groups[g] for w in ws])
        axes[0].scatter(*P2[msk].T, c=c, label=g, s=50)
    for w, (a, b_) in zip(ws, P2):
        axes[0].annotate(w, (a, b_), fontsize=8)
    axes[0].set(title="TinyGPT token embeddings (PCA): exports vs languages", xticks=[], yticks=[])
    axes[0].legend()
    tt["tokens per held-out line"].plot.bar(ax=axes[1], rot=0, color=["C2", "C3", "C0"])
    axes[1].set(ylabel="tokens per line", title="Sequence length by tokenizer")
    save_fig(fig, "10_tokens_and_embeddings")

banner("KEY TAKEAWAYS")
print("""  1. An LLM is a next-token classifier over its vocabulary, applied repeatedly: Modules 3-6 at scale.
  2. Tokenization: BPE learns subword pieces; common words = 1 token, rare words = several, nothing is unknown.
  3. Embeddings: a learned lookup table; words used in similar contexts get similar vectors.
  4. Tokenization leaks into behaviour: spaces, casing, digits, letter counting, and cost per language.
  5. Vocabulary size trades sequence length against table size and data per token.""")
