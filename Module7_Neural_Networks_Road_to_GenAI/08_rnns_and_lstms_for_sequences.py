"""
LAB 08 - RNNs and LSTMs for Sequences
=====================================

THE PROBLEM
    Text, speech, clicks and sensor readings are SEQUENCES: order carries meaning ("dog bites man" vs "man
    bites dog"), and lengths vary. A feed-forward network wants a fixed-size vector; a bag of words throws
    order away.

THE RECURRENT NETWORK (RNN)
    Read one token at a time and keep a running memory, the HIDDEN STATE:
        h_t = tanh(x_t Wx + h_{t-1} Wh + b)          the SAME weights at every step (weight sharing in time)
    The final h summarises everything read so far. Like a CNN shares a filter across positions in space,
    an RNN shares its update across positions in time.

THE LSTM (1997)
    Backprop through time multiplies by Wh and tanh' once per step: over long sequences the gradient vanishes
    (Lab 04, now across time). The LSTM adds a separate CELL STATE with gates (forget, input, output) that
    decide what to keep; information can flow along it for hundreds of steps almost unchanged.
    LSTMs powered machine translation, speech recognition and phone keyboards until ~2017.

LEVELS
    EASY   : An RNN unrolled by hand: the hidden state as memory; parameters don't grow with length.
    MEDIUM : "Does A come before B?": a bag of tokens can't tell; an RNN and an LSTM can.
    HARD   : "Remember the first token": the plain RNN forgets as sequences grow; the LSTM remembers. And the
             catch that led to transformers: recurrence is sequential, so it's slow.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from ml_lab_utils import HAS_MPL, banner, make_sequence_task, note, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, rnn_forward

if not HAS_TORCH:
    raise SystemExit("This lab needs PyTorch:  pip install torch")
import torch
import torch.nn as nn

from nn_toolkit import n_trainable, torch_accuracy, train_classifier

torch.set_num_threads(4)

# %% =========================================================================
# LEVEL 1 (EASY): An RNN by hand
# ============================================================================
banner("A recurrent network, one step at a time", "EASY")
rng = np.random.default_rng(0)
vocab, hidden = 4, 3
Wx, Wh, b = rng.normal(0, 0.8, (vocab, hidden)), rng.normal(0, 0.8, (hidden, hidden)), np.zeros(hidden)
words = ["the", "dog", "bites", "man"]
onehot = np.eye(vocab)
for sentence in (["the", "dog", "bites", "man"], ["the", "man", "bites", "dog"]):
    H = rnn_forward(onehot[[words.index(w) for w in sentence]], Wx, Wh, b)
    print(f"\n  '{' '.join(sentence)}'")
    for w, h in zip(sentence, H):
        print(f"    after {w!r:<8} h = {np.round(h, 3)}")
note("""Both sentences contain the same four words, but the final hidden states
differ: the RNN's memory depends on ORDER, because each step mixes the new word
into what came before. A bag of words would give identical vectors.""")
show("Parameters (Wx, Wh, b) for any sentence length", Wx.size + Wh.size + b.size)
note(f"""The same Wx and Wh are used at every step, so a 4-word and a 4,000-word input
use the same {Wx.size + Wh.size + b.size} numbers. That's how an RNN handles any length.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Order matters
# ============================================================================
banner("Does marker A come before marker B?", "MEDIUM")
S, sy = make_sequence_task("order", n=4000, length=20)
s_tr, s_te = train_test_split(np.arange(len(S)), test_size=0.3, random_state=0)


class SeqClassifier(nn.Module):
    """Embedding -> RNN or LSTM -> last hidden state -> classes.
    forget_bias: start the LSTM's forget gate "open" (keep the memory) instead of half-open."""

    def __init__(self, kind="lstm", vocab=10, emb=16, hidden=64, classes=2, forget_bias=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = (nn.LSTM if kind == "lstm" else nn.RNN)(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, classes)
        if kind == "lstm" and forget_bias:
            for name, prm in self.rnn.named_parameters():
                if "bias" in name:                                   # PyTorch gate order: input, FORGET, cell, output
                    prm.data[hidden:2 * hidden] = forget_bias / 2    # two bias vectors (ih, hh) add up

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h[:, -1])


counts = np.stack([np.bincount(r, minlength=10) for r in S])
rows = {"bag of tokens (gradient boosting on counts)": {
    "test accuracy": HistGradientBoostingClassifier(random_state=0).fit(counts[s_tr], sy[s_tr]).score(counts[s_te], sy[s_te]),
    "parameters": np.nan}}
St, syt = torch.tensor(S), torch.tensor(sy)
for kind in ["rnn", "lstm"]:
    torch.manual_seed(0)
    m = SeqClassifier(kind)
    train_classifier(m, St[s_tr], syt[s_tr], epochs=15, lr=3e-3)
    rows[f"{kind.upper()} (reads in order)"] = {"test accuracy": torch_accuracy(m, St[s_te], syt[s_te]), "parameters": n_trainable(m)}
order_tab = pd.DataFrame(rows).T
print(order_tab.to_string(formatters={"test accuracy": "{:.3f}".format, "parameters": lambda v: "-" if np.isnan(v) else f"{v:,.0f}"}))
note(f"""Counts are identical for both classes, so the bag of tokens is a coin flip
({order_tab.iloc[0, 0]:.0%}). Reading in order, both recurrent networks learn the rule, from
examples alone: 'remember whether you've seen A; when B arrives, check'.
Nobody told them which tokens matter or where they are.""")

# %% =========================================================================
# LEVEL 3 (HARD): Long memory, and the speed problem
# ============================================================================
banner("Remember the first token: RNN vs LSTM as sequences grow", "HARD")
rows = {}
VARIANTS = {"RNN": ("rnn", 0.0), "LSTM, default init": ("lstm", 0.0), "LSTM, forget bias 4": ("lstm", 4.0)}
for length in [10, 30, 60, 100]:
    R, ry = make_sequence_task("remember", n=4000, length=length)
    Rt, ryt = torch.tensor(R), torch.tensor(ry)
    for name, (kind, fb) in VARIANTS.items():
        torch.manual_seed(0)
        m = SeqClassifier(kind, classes=4, forget_bias=fb)
        t0 = time.perf_counter()
        hist = train_classifier(m, Rt[s_tr], ryt[s_tr], Rt[s_te], ryt[s_te], epochs=15, lr=3e-3)
        rows.setdefault(length, {})[name] = hist[-1]["test_acc"]
        rows[length][f"{name} seconds"] = time.perf_counter() - t0
mem = pd.DataFrame(rows).T
mem.index.name = "sequence length"
print(mem[list(VARIANTS)].round(3).to_string())
note(f"""The label is the FIRST token; everything after it is noise; guessing scores 25%.
At 10 steps everyone remembers. At 100 steps the plain RNN is back to guessing
({mem.loc[100, 'RNN']:.0%}): the gradient from the answer has to travel back through 100
multiplications and fades away (Lab 04's vanishing gradient, in time).
Plot twist: the LSTM with PyTorch's DEFAULT initialisation fails already at 30
steps ({mem.loc[30, 'LSTM, default init']:.0%}). Its forget gate starts half-open, so the memory decays by
~half per step until training learns to keep it, and here it never does.
Start the gate open (bias 4, a standard trick) and the same architecture
remembers across all 100 steps ({mem.loc[100, 'LSTM, forget bias 4']:.0%}). The gated cell state CAN carry
information; the initialisation decides whether training finds out.""")

step("3.1  How much does the final answer depend on each input position?")
R, ry = make_sequence_task("remember", n=600, length=60)
sens = {}
for name, (kind, fb) in VARIANTS.items():
    torch.manual_seed(0)
    m = SeqClassifier(kind, classes=4, forget_bias=fb)
    train_classifier(m, torch.tensor(R[:500]), torch.tensor(ry[:500]), epochs=1, lr=3e-3)
    e = m.emb(torch.tensor(R[500:])).detach().requires_grad_(True)
    h, _ = m.rnn(e)
    nn.functional.cross_entropy(m.out(h[:, -1]), torch.tensor(ry[500:])).backward()
    g = e.grad.norm(dim=-1).mean(0).numpy()
    sens[name] = g / g[-1]
sens_tab = pd.DataFrame(sens, index=[f"position {p}" for p in range(60)]).iloc[[0, 10, 30, 50, 59]]
print(sens_tab.to_string(float_format=lambda v: f"{v:.1e}"))
note(f"""Gradient of the loss with respect to each input, relative to the last
position, early in training. For the plain RNN and the default LSTM, the first
position (the one that holds the answer!) gets ~{sens_tab.iloc[0, :2].max():.0e} of the signal the last
position gets: it can hardly learn to use it. With the forget gate open,
position 0 gets {sens_tab.iloc[0, 2]:.1f}x the signal of the last position: the memory path is clear.""")

step("3.2  The catch: recurrence is sequential")
note(f"""Step t needs h_(t-1), so the steps can't run in parallel: training time grows
with length (LSTM: {mem.loc[10, 'LSTM, forget bias 4 seconds']:.1f}s at 10 steps, {mem.loc[100, 'LSTM, forget bias 4 seconds']:.1f}s at 100), and a GPU's thousands of
cores sit idle. Memory is also squeezed through one fixed-size vector: to
translate word 50 the model must recall word 3 from a 64-number summary. In
2017 'Attention Is All You Need' dropped recurrence: every position looks at
every other position directly, all at once. That is Lab 09.""", keep_lines=False)

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for name in VARIANTS:
        axes[0].plot(mem.index, mem[name], "o-", label=name)
    axes[0].axhline(0.25, color="grey", ls=":", label="guessing")
    axes[0].set(xlabel="sequence length", ylabel="test accuracy", title="Remember the first token", ylim=(0, 1.05))
    axes[0].legend()
    full = pd.DataFrame(sens, index=range(60))
    for k in full:
        axes[1].semilogy(full.index, full[k], label=k)
    axes[1].set(xlabel="input position (0 = the answer)", ylabel="gradient size (relative to last)",
                title="How far back does the learning signal reach?")
    axes[1].legend()
    for name in VARIANTS:
        axes[2].plot(mem.index, mem[f"{name} seconds"], "o-", label=name)
    axes[2].set(xlabel="sequence length", ylabel="training seconds (15 epochs)", title="Sequential = slower as length grows")
    axes[2].legend()
    save_fig(fig, "08_rnns_and_lstms_for_sequences")

banner("KEY TAKEAWAYS")
print("""  1. An RNN reads one token at a time and carries a hidden state: order matters, any length works.
  2. The same weights are reused at every step, so parameters don't grow with sequence length.
  3. Plain RNNs forget over long gaps: the gradient vanishes through time.
  4. LSTMs add a gated cell state that carries information across many steps, if initialised to keep it.
  5. Recurrence is sequential (slow) and squeezes memory into one vector: attention removes both limits.""")
