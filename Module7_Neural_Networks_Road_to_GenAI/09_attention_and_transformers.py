"""
LAB 09 - Attention, the Transformer, Encoders and Decoders
==========================================================

ATTENTION: a soft lookup
    Every position asks a question (QUERY), every position advertises what it has (KEY) and what it would
    hand over (VALUE). Position i's output = a weighted average of all values, weighted by how well its query
    matches each key:
        weights = softmax(Q K^T / sqrt(d))        output = weights @ V
    Any position can read any other in ONE step: no memory squeezed through a hidden state, no waiting for
    the previous step. All positions are computed at once (in parallel, on a GPU).

THE TRANSFORMER BLOCK (2017)
    x = x + MultiHeadAttention(LayerNorm(x))      positions exchange information
    x = x + MLP(LayerNorm(x))                     each position processes what it gathered
    + positional information (attention alone ignores order) + residual connections (Lab 04) everywhere.
    GPT-3 = 96 of these blocks. Nothing else.

ENCODER vs DECODER
    encoder (BERT)          every position sees every position     understand: classify, embed, search (Module 6)
    decoder (GPT, Claude)   each position sees only the PAST        generate: predict the next token, repeat
    encoder-decoder (T5)    an encoder reads the input; a decoder writes the output while attending to it
                            (translation, the original 2017 design)

LEVELS
    EASY   : Attention by hand: "it" finds "cat". The causal mask. Attention without positions ignores order.
    MEDIUM : A tiny transformer on Lab 08's tasks: memory across 100 steps, order with and without positions.
    HARD   : Encoder vs decoder, measured: what each position is allowed to see. Anatomy of a block.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_lab_utils import HAS_MPL, banner, make_sequence_task, note, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, attention, causal_mask, sinusoidal_positions

if not HAS_TORCH:
    raise SystemExit("This lab needs PyTorch:  pip install torch")
import torch
import torch.nn as nn

from nn_toolkit import TransformerBlock, n_trainable, torch_accuracy, train_classifier

torch.set_num_threads(4)

# %% =========================================================================
# LEVEL 1 (EASY): Attention by hand
# ============================================================================
banner("Attention by hand: who does 'it' look at?", "EASY")
words = ["the", "cat", "sat", "because", "it", "was", "tired"]
#                    animal  thing-to-refer-to  pronoun  verb
feats = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.5, 0, 0, 0.3]], float)
Wq = np.zeros((4, 4))
Wq[2, 1] = 3.0                                                       # a pronoun's QUERY: "I'm looking for a referent"
Wk = np.eye(4) * 3.0                                                 # every word's KEY: its own features
Q, K, V = feats @ Wq, feats @ Wk, feats
out, w = attention(Q, K, V)
print(pd.DataFrame(w, index=words, columns=words).round(2).loc[["it", "cat", "tired"]].to_string())
show("Output for 'it' (= mostly the features of 'cat')", out[words.index("it")].round(2).tolist())
note("""The pronoun's query matches the key of 'cat' (a referent), so 'it' puts most of
its attention on 'cat' and its output vector becomes mostly cat's features:
the model now 'knows' what 'it' refers to. Words whose query matches nothing
in particular spread their attention evenly. In a trained model nobody writes
Wq and Wk; they are learned, with dozens of heads, each asking a different
question (grammar, coreference, nearby words, ...).""")

step("1.1  The causal mask: a decoder can't look ahead")
_, wc = attention(Q, K, V, mask=causal_mask(len(words)))
print(pd.DataFrame(wc, index=words, columns=words).round(2).loc[["the", "it", "tired"]].to_string())
note("""With the causal mask, position i may only attend to positions 0..i (the upper
triangle is set to -infinity before the softmax). 'it' still finds 'cat'
(which came earlier), but 'the' can only look at itself. This is what lets
GPT train on 'predict the next word' at every position at once without
cheating: no position can peek at the word it must predict.""")

step("1.2  Attention alone ignores word order")
perm = np.array([4, 1, 6, 0, 2, 5, 3])                               # shuffle the sentence
out_p, _ = attention(Q[perm], K[perm], V[perm])
show("Shuffled sentence: same outputs, just shuffled?", bool(np.allclose(out_p, out[perm])))
P = sinusoidal_positions(len(words), 4)                              # one vector per SLOT, not per word
with_pos = feats + P
shuffled_pos = feats[perm] + P                                       # words move, slot positions stay
same = np.allclose(attention(shuffled_pos @ Wq, shuffled_pos @ Wk, shuffled_pos)[0],
                   attention(with_pos @ Wq, with_pos @ Wk, with_pos)[0][perm])
show("Add position vectors: still the same?", same)
note("""Attention is a weighted average over a SET: shuffle the words and each output
is unchanged, so 'dog bites man' = 'man bites dog'. Transformers therefore ADD
a position vector to every token embedding (sinusoids in 2017, learned
vectors in GPT-2 and our TinyGPT, rotations in most current LLMs). The
position vectors stay with their slots when the words move, so a shuffled
sentence now produces different outputs.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): A tiny transformer on Lab 08's tasks
# ============================================================================
banner("A tiny transformer vs Lab 08's tasks", "MEDIUM")


class TinyTransformer(nn.Module):
    """token embedding (+ position embedding) -> blocks -> read one position (or the mean) -> classes."""

    def __init__(self, vocab=10, d=64, heads=4, layers=2, context=128, classes=2, positions=True, causal=False, read="mean"):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(context, d) if positions else None
        self.blocks = nn.ModuleList([TransformerBlock(d, heads, 0.0, causal=causal) for _ in range(layers)])
        self.ln, self.out, self.read = nn.LayerNorm(d), nn.Linear(d, classes), read

    def forward(self, x, keep_attn=False):
        h = self.tok(x)
        if self.pos is not None:
            h = h + self.pos(torch.arange(x.shape[1]))
        for blk in self.blocks:
            h = blk(h, keep_attn)
        h = self.ln(h)
        return self.out(h.mean(1) if self.read == "mean" else h[:, 0] if self.read == "first" else h[:, -1])


idx_tr, idx_te = train_test_split(np.arange(4000), test_size=0.3, random_state=0)


def run(model, X, y, epochs=15, lr=2e-3):
    X, y = torch.tensor(X), torch.tensor(y)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    hist = train_classifier(model, X[idx_tr], y[idx_tr], epochs=epochs, lr=lr)
    return {"test accuracy": torch_accuracy(model, X[idx_te], y[idx_te]), "seconds": time.perf_counter() - t0,
            "parameters": n_trainable(model)}


step("2.1  Remember the first token across 100 steps")
R, ry = make_sequence_task("remember", n=4000, length=100)
torch.manual_seed(0)
mem_model = TinyTransformer(classes=4)
mem = run(mem_model, R, ry)
show("Transformer: test accuracy / seconds", f"{mem['test accuracy']:.3f} / {mem['seconds']:.1f}s")
with torch.no_grad():
    mem_model(torch.tensor(R[idx_te[:200]]), keep_attn=True)
att_first = mem_model.blocks[0].last_attn.mean(dim=(0, 1))          # average over examples and heads: (T, T)
show("Share of attention that lands on position 0", f"{att_first[:, 0].mean():.2f}   (uniform would be {1 / 100:.2f})")
note(f"""Lab 08's plain RNN was back to guessing at 100 steps, and the LSTM needed an
initialisation trick. The transformer solves it directly: every position can
look at position 0 in one hop, no matter how far away, and the first layer
learns to do exactly that. The distance between two tokens no longer matters.
Honest footnote: on this laptop CPU it took {mem['seconds']:.0f}s, slower than Lab 08's LSTM.
Attention compares every position with every other (100 x 100 scores per head
per layer); the payoff is that all of them run at once, which GPUs exploit
and a CPU barely can. That T x T cost is also why long context windows are
expensive.""")

step("2.2  Order: does A come before B? With and without positions")
S, sy = make_sequence_task("order", n=4000, length=20)
rows = {"with position embeddings": run(TinyTransformer(positions=True), S, sy),
        "WITHOUT position embeddings": run(TinyTransformer(positions=False), S, sy)}
ordtab = pd.DataFrame(rows).T
print(ordtab.to_string(formatters={"test accuracy": "{:.3f}".format, "seconds": "{:.1f}".format, "parameters": "{:,.0f}".format}))
note(f"""Without position information the transformer is a very elaborate bag of
words: it sees one A and one B in both classes and sits at a coin flip
({ordtab.iloc[1, 0]:.0%}). Add position embeddings and it solves the task ({ordtab.iloc[0, 0]:.0%}). The
architecture's power comes from attention; its sense of order comes entirely
from the position vectors.""")

# %% =========================================================================
# LEVEL 3 (HARD): Encoder vs decoder, and the anatomy of a block
# ============================================================================
banner("Encoder vs decoder: what each position is allowed to see", "HARD")
step("3.1  The answer is in the LAST token. Read it from the first or the last position")
Rl = np.ascontiguousarray(R[:, ::-1][:, -30:])                       # 30 tokens, the label token is now LAST
rows = {}
for kind, causal in [("encoder (bidirectional)", False), ("decoder (causal)", True)]:
    for read in ["first", "last"]:
        torch.manual_seed(0)
        rows.setdefault(kind, {})[f"read position: {read}"] = run(TinyTransformer(classes=4, causal=causal, read=read), Rl, ry)["test accuracy"]
encdec = pd.DataFrame(rows).T
print(encdec.round(3).to_string())
note(f"""The encoder can read the answer from ANY position: every position sees the
whole sequence. The decoder's first position sees only itself, so reading
from there is guessing ({encdec.iloc[1, 0]:.0%}); its LAST position has seen everything and
works. That's why GPT-style models classify (and answer) from the final
token, while BERT-style encoders produce a context-aware vector for every
token, ideal for search embeddings (Module 6) and classification.""")

step("3.2  Proof: in a decoder, the future can't change the past")
torch.manual_seed(0)
blk_dec, blk_enc = TransformerBlock(32, 4, 0.0, causal=True).eval(), TransformerBlock(32, 4, 0.0, causal=False).eval()
x = torch.randn(1, 6, 32)
x2 = x.clone()
x2[0, -1] = torch.randn(32)                                          # change only the LAST token
with torch.no_grad():
    d_dec = (blk_dec(x) - blk_dec(x2))[0].norm(dim=-1)
    d_enc = (blk_enc(x) - blk_enc(x2))[0].norm(dim=-1)
print(pd.DataFrame({"decoder": d_dec.numpy(), "encoder": d_enc.numpy()},
                   index=[f"position {i}" for i in range(6)]).round(4).T.to_string())
note("""Change only the last token: in the decoder, positions 0-4 are bit-for-bit
unchanged; in the encoder every position changes. This is why a decoder can
GENERATE: the representations of the words already written never depend on
words not yet written, so they can be computed once and cached (the 'KV
cache' that makes chatbots fast).""")

step("3.3  Anatomy of one block, and the road to GPT")
blk = TransformerBlock(768, 12, 0.0)
parts = {"attention: Q, K, V projections": blk.qkv, "attention: output projection": blk.proj, "MLP (768 -> 3072 -> 768)": blk.mlp,
         "two LayerNorms": nn.ModuleList([blk.ln1, blk.ln2])}
anat = pd.Series({k: n_trainable(v) for k, v in parts.items()})
anat["one block, total"] = anat.sum()
print(anat.map("{:,}".format).to_string())
show("x 12 blocks + token/position embeddings (GPT-2 small)", f"~{(12 * anat['one block, total'] + 50_257 * 768 + 1024 * 768) / 1e6:.0f} million")
note("""Two thirds of every block is the MLP: attention MOVES information between
positions, the MLP TRANSFORMS it at each position (and is where much of a
model's factual knowledge seems to be stored). GPT-2 small is 12 of these
blocks, 768 wide; GPT-3 is 96 blocks, 12,288 wide. Same recipe, bigger
numbers. In Lab 11 we train a 4-block version on our invented world.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].imshow(w, cmap="Blues")
    axes[0].set(xticks=range(7), yticks=range(7), xticklabels=words, yticklabels=words, title="Hand-made attention (rows look at columns)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].imshow(att_first[:30, :30].numpy(), cmap="Blues")
    axes[1].set(title="Learned attention, 'remember' task (first 30 positions)", xlabel="attends to position", ylabel="position")
    encdec.plot.bar(ax=axes[2], rot=0)
    axes[2].axhline(0.25, color="grey", ls=":")
    axes[2].set(title="The answer is in the last token", ylim=(0, 1.05))
    save_fig(fig, "09_attention_and_transformers")

banner("KEY TAKEAWAYS")
print("""  1. Attention = softmax(Q K^T / sqrt(d)) V: each position takes a weighted average of what others offer.
  2. Any two positions are one hop apart, and all positions run in parallel: no RNN bottleneck.
  3. Attention ignores order by itself; position embeddings supply it.
  4. A transformer block = attention (move information) + MLP (transform it), with residuals and LayerNorm.
  5. Encoders see everything (understand, embed); decoders see only the past (generate, cache).""")
