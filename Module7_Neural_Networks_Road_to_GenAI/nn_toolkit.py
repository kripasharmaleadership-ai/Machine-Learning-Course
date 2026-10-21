"""
nn_toolkit.py - The neural-network library we build across Module 7.

    1. NumPy, from scratch (Labs 02-06): every number visible
       sigmoid / tanh / relu / softmax      activation functions and their derivatives
       bce_with_logits / softmax_ce / mse   loss functions (numerically stable) and their gradients
       Perceptron                            Rosenblatt's 1958 learning rule
       MLP                                   a feed-forward network: forward, backward, dropout, any optimizer
       numerical_gradient()                  finite differences: how we PROVE backprop is right
       SGD / Momentum / Adam                 optimizers
    2. NumPy building blocks of modern architectures (Labs 07-09)
       conv2d()                              a convolution, the CNN's only new idea
       rnn_forward()                         a recurrent network unrolled over time
       attention()                           scaled dot-product attention, with masks
       sinusoidal_positions(), causal_mask()
       train_classifier / torch_accuracy     the standard PyTorch training loop (Labs 07-09, 14)
    3. Language models (Labs 10-13)
       BPETokenizer                          byte-pair encoding, trained on our corpus (GPT-style)
       TransformerBlock (PyTorch)            attention + MLP + residuals + LayerNorm; causal (decoder) or not (encoder)
       TinyGPT (PyTorch)                     a decoder-only transformer that exposes its attention maps
       train_lm_stream / train_lm_examples   pretraining on a text stream; instruction-style fine-tuning
       next_token_probs / generate           temperature, top-k, top-p sampling
       continuation_logprob                  how likely is this answer? (probing knowledge without sampling)
       get_or_train()                        cache trained models in outputs/ so later labs start instantly
    4. ModelScorecard (as in Modules 3-6)
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. NumPy from scratch: activations and losses
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))                              # = 1/(1+e^-z), without overflow


def relu(z):
    return np.maximum(0, z)


ACTIVATIONS = {   # name: (function, derivative expressed with the PRE-activation z)
    "sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
    "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    "relu": (relu, lambda z: (z > 0).astype(float)),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)                          # the stability trick: subtract the max
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def bce_with_logits(logits, y):
    """Binary cross-entropy from raw scores, stable for any logit. Returns (mean loss, d loss / d logits)."""
    logits, y = logits.ravel(), y.ravel()
    loss = np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return loss.mean(), ((sigmoid(logits) - y) / len(y)).reshape(-1, 1)


def softmax_ce(logits, y):
    """Multi-class cross-entropy. y: integer labels. Returns (mean loss, gradient w.r.t. logits)."""
    n = len(y)
    logz = logits.max(axis=1, keepdims=True)
    logz = logz + np.log(np.exp(logits - logz).sum(axis=1, keepdims=True))
    loss = (logz.ravel() - logits[np.arange(n), y]).mean()
    grad = softmax(logits)
    grad[np.arange(n), y] -= 1
    return loss, grad / n


def mse(pred, y):
    pred, y = pred.ravel(), y.ravel()
    return float(np.mean((pred - y) ** 2)), (2 * (pred - y) / len(y)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# 1b. The perceptron and the multi-layer perceptron
# ---------------------------------------------------------------------------
class Perceptron:
    """Rosenblatt's perceptron: predict sign(w.x + b); on a mistake, w += lr * y * x. Labels in {-1, +1}."""

    def __init__(self, lr: float = 1.0, epochs: int = 20):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.w, self.b, self.mistakes_ = np.zeros(X.shape[1]), 0.0, []
        for _ in range(self.epochs):
            m = 0
            for xi, yi in zip(X, y):
                if yi * (xi @ self.w + self.b) <= 0:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    m += 1
            self.mistakes_.append(m)
            if m == 0:
                break
        return self

    def predict(self, X):
        return np.where(X @ self.w + self.b > 0, 1, -1)


class SGD:
    def __init__(self, lr=0.1, momentum=0.0):
        self.lr, self.momentum, self.v = lr, momentum, None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.momentum
            v -= self.lr * g
            p += v


def Momentum(lr=0.05, momentum=0.9):
    return SGD(lr, momentum)


class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t, self.m, self.v = lr, b1, b2, eps, 0, None, None

    def step(self, params, grads):
        if self.m is None:
            self.m, self.v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class MLP:
    """A feed-forward network in NumPy.
        sizes      : e.g. [2, 16, 16, 1]  (inputs, hidden layers..., outputs)
        activation : hidden activation ('relu', 'tanh', 'sigmoid', 'linear')
        loss       : 'bce' (1 output), 'ce' (k outputs, integer labels) or 'mse'
        dropout    : probability of zeroing each hidden unit during training (inverted dropout)
        init       : 'he' (relu), 'xavier' (tanh/sigmoid), or a float standard deviation"""

    def __init__(self, sizes, activation="relu", loss="bce", dropout=0.0, init="auto", l2=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes, self.act_name, self.loss_name, self.dropout, self.l2 = sizes, activation, loss, dropout, l2
        self.rng = np.random.default_rng(seed + 1)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            if init == "auto":
                std = np.sqrt(2 / fan_in) if activation == "relu" else np.sqrt(1 / fan_in)
            elif init == "he":
                std = np.sqrt(2 / fan_in)
            elif init == "xavier":
                std = np.sqrt(1 / fan_in)
            else:
                std = float(init)
            self.W.append(rng.normal(0, std, (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    @property
    def params(self):
        return self.W + self.b

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, X, train=False):
        f = ACTIVATIONS[self.act_name][0]
        self.cache = [(X, None, None)]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i == len(self.W) - 1:
                self.cache.append((z, z, None))
                return z
            h = f(z)
            mask = None
            if train and self.dropout > 0:
                mask = (self.rng.random(h.shape) >= self.dropout) / (1 - self.dropout)
                h = h * mask
            self.cache.append((h, z, mask))
        return h

    def loss_and_grad(self, out, y):
        if self.loss_name == "bce":
            return bce_with_logits(out, y)
        if self.loss_name == "ce":
            return softmax_ce(out, y)
        return mse(out, y)

    def backward(self, dout):
        """Backpropagation: walk the cached layers in reverse, applying the chain rule."""
        dfn = ACTIVATIONS[self.act_name][1]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        delta = dout
        for i in reversed(range(len(self.W))):
            h_prev = self.cache[i][0]
            gW[i] = h_prev.T @ delta + self.l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i > 0:
                _, z_prev, mask = self.cache[i]
                delta = delta @ self.W[i].T
                if mask is not None:
                    delta = delta * mask
                delta = delta * dfn(z_prev)
        self.grad_norms_ = [float(np.linalg.norm(g)) for g in gW]
        return gW + gb

    def fit(self, X, y, epochs=100, batch_size=32, optimizer=None, X_val=None, y_val=None, seed=0, verbose=False):
        opt = optimizer or Adam(0.01)
        rng = np.random.default_rng(seed)
        self.history_ = []
        for ep in range(epochs):
            order = rng.permutation(len(X))
            losses = []
            for s in range(0, len(X), batch_size):
                idx = order[s:s + batch_size]
                out = self.forward(X[idx], train=True)
                loss, d = self.loss_and_grad(out, y[idx])
                opt.step(self.params, self.backward(d))
                losses.append(loss)
            row = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "train_acc": self.score(X, y)}
            if X_val is not None:
                row.update(val_loss=float(self.loss_and_grad(self.forward(X_val), y_val)[0]), val_acc=self.score(X_val, y_val))
            self.history_.append(row)
            if verbose and (ep + 1) % verbose == 0:
                print("   ", {k: round(v, 4) for k, v in row.items()})
        return self

    def predict_proba(self, X):
        out = self.forward(X)
        return sigmoid(out).ravel() if self.loss_name == "bce" else softmax(out) if self.loss_name == "ce" else out.ravel()

    def predict(self, X):
        p = self.predict_proba(X)
        return (p > 0.5).astype(int) if self.loss_name == "bce" else p.argmax(axis=1) if self.loss_name == "ce" else p

    def score(self, X, y):
        if self.loss_name == "mse":
            return float(-np.mean((self.predict(X) - y) ** 2))
        return float(np.mean(self.predict(X) == y))


def numerical_gradient(f: Callable[[], float], params, eps: float = 1e-5):
    """Finite differences: nudge every parameter up and down and watch the loss. Slow, but certainly right."""
    grads = []
    for p in params:
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            up = f()
            p[i] = old - eps
            down = f()
            p[i] = old
            g[i] = (up - down) / (2 * eps)
        grads.append(g)
    return grads


# ---------------------------------------------------------------------------
# 2. Building blocks of CNNs, RNNs and transformers (NumPy)
# ---------------------------------------------------------------------------
def conv2d(image, kernel, stride: int = 1):
    """'Valid' 2-D convolution (cross-correlation, as in deep learning): slide the kernel, multiply, sum."""
    kh, kw = kernel.shape
    H = (image.shape[0] - kh) // stride + 1
    W = (image.shape[1] - kw) // stride + 1
    out = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            out[r, c] = np.sum(image[r * stride:r * stride + kh, c * stride:c * stride + kw] * kernel)
    return out


def max_pool(x, size: int = 2):
    H, W = x.shape[0] // size, x.shape[1] // size
    return x[:H * size, :W * size].reshape(H, size, W, size).max(axis=(1, 3))


def rnn_forward(X_onehot, Wx, Wh, b, h0=None):
    """A vanilla RNN unrolled over time: h_t = tanh(x_t Wx + h_{t-1} Wh + b). X: (T, d). Returns all h (T, hidden)."""
    h = np.zeros(Wh.shape[0]) if h0 is None else h0
    hs = []
    for x in X_onehot:
        h = np.tanh(x @ Wx + h @ Wh + b)
        hs.append(h)
    return np.array(hs)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention: softmax(Q K^T / sqrt(d)) V. mask: True where attention is NOT allowed.
    Returns (output, weights)."""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    w = softmax(scores, axis=-1)
    return w @ V, w


def causal_mask(T: int):
    """True above the diagonal: position t may look at positions <= t only (a decoder, like GPT)."""
    return np.triu(np.ones((T, T), dtype=bool), k=1)


def sinusoidal_positions(T: int, d: int):
    pos, i = np.arange(T)[:, None], np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))


# ---------------------------------------------------------------------------
# 3. Tokenisation: byte-pair encoding
# ---------------------------------------------------------------------------
_PRETOK = re.compile(r" ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]+|\s+")


class BPETokenizer:
    """GPT-style byte-pair encoding (on characters).
    Text is first split into chunks (a word with its leading space, a digit, punctuation). Training starts from
    single characters and repeatedly MERGES the most frequent adjacent pair into a new token, vocab_size times.
    Frequent words become one token; rare words are spelled with several pieces; nothing is ever unknown
    if its characters were seen."""
    UNK = "<unk>"

    def train(self, text: str, vocab_size: int = 400, verbose: bool = False):
        chunks = Counter(_PRETOK.findall(text))
        words = {w: list(w) for w in chunks}
        chars = sorted({ch for w in chunks for ch in w})
        self.vocab = [self.UNK] + chars
        self.merges = {}
        while len(self.vocab) < vocab_size:
            pairs = Counter()
            for w, freq in chunks.items():
                sym = words[w]
                for a, b in zip(sym, sym[1:]):
                    pairs[a, b] += freq
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            if freq < 2:
                break
            new = a + b
            self.merges[a, b] = len(self.merges)
            self.vocab.append(new)
            if verbose and len(self.merges) <= verbose:
                print(f"    merge {len(self.merges):>3}: {a!r} + {b!r} -> {new!r}   (seen {freq} times)")
            for w in words:
                sym, out, i = words[w], [], 0
                while i < len(sym):
                    if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(sym[i])
                        i += 1
                words[w] = out
        self.id_of = {t: i for i, t in enumerate(self.vocab)}
        self._cache = {}
        return self

    def _encode_chunk(self, chunk: str) -> list[str]:
        if chunk in self._cache:
            return self._cache[chunk]
        sym = list(chunk)
        while len(sym) > 1:
            ranks = [(self.merges.get((a, b), 1e18), i) for i, (a, b) in enumerate(zip(sym, sym[1:]))]
            r, i = min(ranks)
            if r == 1e18:
                break
            sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
        self._cache[chunk] = sym
        return sym

    def tokens(self, text: str) -> list[str]:
        return [t for ch in _PRETOK.findall(text) for t in self._encode_chunk(ch)]

    def encode(self, text: str) -> list[int]:
        return [self.id_of.get(t, 0) for t in self.tokens(text)]

    def decode(self, ids) -> str:
        return "".join(self.vocab[i] if i > 0 else "?" for i in ids)

    def to_dict(self):
        return {"vocab": self.vocab, "merges": [[a, b] for (a, b) in self.merges]}

    @classmethod
    def from_dict(cls, d):
        t = cls()
        t.vocab = d["vocab"]
        t.merges = {(a, b): i for i, (a, b) in enumerate(d["merges"])}
        t.id_of = {tok: i for i, tok in enumerate(t.vocab)}
        t._cache = {}
        return t


# ---------------------------------------------------------------------------
# 3b. A tiny GPT in PyTorch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:                                                  # Labs 01-06 work without PyTorch
    HAS_TORCH = False

if HAS_TORCH:
    class TransformerBlock(nn.Module):
        """One transformer block: x + Attention(LayerNorm(x)), then x + MLP(LayerNorm(x)).
        causal=True: a DECODER block (GPT), each position sees only itself and the past.
        causal=False: an ENCODER block (BERT), every position sees every other."""

        def __init__(self, d, heads, dropout, causal=True):
            super().__init__()
            self.heads, self.causal = heads, causal
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.drop = nn.Dropout(dropout)
            self.last_attn = None

        def forward(self, x, keep_attn=False):
            B, T, D = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
            q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
            att = (q @ k.transpose(-2, -1)) / (D // self.heads) ** 0.5
            if self.causal:
                att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), float("-inf"))
            att = att.softmax(dim=-1)
            if keep_attn:
                self.last_attn = att.detach()
            y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, D)
            x = x + self.drop(self.proj(y))
            return x + self.drop(self.mlp(self.ln2(x)))

    class TinyGPT(nn.Module):
        """A decoder-only transformer (the GPT recipe), small enough to train on a laptop CPU:
        token embeddings + learned position embeddings -> n_layers x [causal self-attention + MLP] -> logits."""

        def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, context=64, dropout=0.1):
            super().__init__()
            self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                               context=context, dropout=dropout)
            self.context = context
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(context, d_model)
            self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok.weight                       # weight tying (as in GPT-2)
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

        def forward(self, idx, keep_attn=False):
            T = idx.shape[1]
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
            for blk in self.blocks:
                x = blk(x, keep_attn)
            return self.head(self.ln(x))

        @property
        def n_params(self):
            return sum(p.numel() for p in self.parameters())

    def train_classifier(model, X_tr, y_tr, X_te=None, y_te=None, epochs=20, lr=3e-3, batch=64, seed=0, weight_decay=0.0):
        """The standard PyTorch loop (CNNs, RNNs, transformers): AdamW, cross-entropy, mini-batches.
        X: a float or long array/tensor; y: integer labels. Returns a list of per-epoch dicts."""
        torch.manual_seed(seed)
        X_tr, y_tr = torch.as_tensor(X_tr), torch.as_tensor(y_tr, dtype=torch.long)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        g = torch.Generator().manual_seed(seed)
        hist = []
        for ep in range(epochs):
            model.train()
            order = torch.randperm(len(X_tr), generator=g)
            losses = []
            for s in range(0, len(X_tr), batch):
                idx = order[s:s + batch]
                loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            row = {"epoch": ep + 1, "train_loss": float(np.mean(losses))}
            if X_te is not None:
                row["test_acc"] = torch_accuracy(model, X_te, y_te)
            hist.append(row)
        model.eval()
        return hist

    @torch.no_grad()
    def torch_accuracy(model, X, y, batch=512):
        model.eval()
        X, y = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.long)
        pred = torch.cat([model(X[s:s + batch]).argmax(-1) for s in range(0, len(X), batch)])
        return float((pred == y).float().mean())

    def n_trainable(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_lm_stream(model, ids, steps=1500, batch=32, lr=3e-3, seed=0, log_every=0, warmup=100,
                        checkpoints=(), on_checkpoint=None):
        """Pretraining: sample random windows from one long token stream; predict every next token.
        on_checkpoint(step, model) is called (in eval mode) at each step listed in `checkpoints`."""
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        data = torch.as_tensor(ids, dtype=torch.long)
        T = model.context
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
        model.train()
        hist = []
        for s in range(steps):
            ix = torch.randint(0, len(data) - T - 1, (batch,), generator=g)
            x = torch.stack([data[i:i + T] for i in ix])
            y = torch.stack([data[i + 1:i + T + 1] for i in ix])
            loss = F.cross_entropy(model(x).reshape(-1, model.config["vocab_size"]), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            hist.append(loss.item())
            if log_every and (s + 1) % log_every == 0:
                print(f"    step {s + 1:>5}  loss {np.mean(hist[-log_every:]):.3f}")
            if on_checkpoint and (s + 1) in checkpoints:
                model.eval()
                on_checkpoint(s + 1, model)
                model.train()
        model.eval()
        return hist

    def train_lm_examples(model, examples, steps=600, batch=32, lr=1e-3, seed=0, log_every=0):
        """Fine-tuning on (prompt_ids, answer_ids) examples: the loss counts ONLY the answer tokens,
        which is how instruction tuning teaches 'when you see this format, answer like this'."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        model.train()
        hist = []
        T = model.context
        for s in range(steps):
            pick = rng.integers(0, len(examples), batch)
            seqs = [(examples[i][0] + examples[i][1])[-(T + 1):] for i in pick]
            n_ans = [min(len(examples[i][1]), T) for i in pick]
            L = max(len(q) for q in seqs)
            x = torch.zeros(batch, L - 1, dtype=torch.long)
            y = torch.full((batch, L - 1), -100, dtype=torch.long)
            for r, (q, na) in enumerate(zip(seqs, n_ans)):
                x[r, :len(q) - 1] = torch.as_tensor(q[:-1])
                tgt = torch.full((len(q) - 1,), -100, dtype=torch.long)
                tgt[-na:] = torch.as_tensor(q[-na:])
                y[r, :len(q) - 1] = tgt
            loss = F.cross_entropy(model(x).reshape(-1, model.config["vocab_size"]), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            hist.append(loss.item())
            if log_every and (s + 1) % log_every == 0:
                print(f"    step {s + 1:>5}  loss {np.mean(hist[-log_every:]):.3f}")
        model.eval()
        return hist

    @torch.no_grad()
    def next_token_probs(model, tok, prompt: str, temperature: float = 1.0):
        ids = tok.encode(prompt)[-model.context:]
        logits = model(torch.as_tensor([ids]))[0, -1]
        return torch.softmax(logits / max(temperature, 1e-6), dim=-1).numpy()

    @torch.no_grad()
    def generate(model, tok, prompt: str, max_new_tokens=30, temperature=1.0, top_k=None, top_p=None, seed=0,
                 stop: str | None = "\n"):
        """Autoregressive generation: predict a distribution, pick a token, append it, repeat.
        temperature 0 = always the most likely token (greedy). Returns (text, list of (token, probability))."""
        g = torch.Generator().manual_seed(seed)
        ids = tok.encode(prompt)
        picked = []
        for _ in range(max_new_tokens):
            logits = model(torch.as_tensor([ids[-model.context:]]))[0, -1]
            if temperature <= 0:
                nxt = int(logits.argmax())
                p = torch.softmax(logits, -1)
            else:
                p = torch.softmax(logits / temperature, -1)
                if top_k:
                    kth = torch.topk(p, top_k).values[-1]
                    p = torch.where(p >= kth, p, torch.zeros_like(p))
                if top_p:
                    sp, si = torch.sort(p, descending=True)
                    keep = torch.cumsum(sp, 0) - sp < top_p
                    p = torch.zeros_like(p).scatter(0, si[keep], sp[keep])
                p = p / p.sum()
                nxt = int(torch.multinomial(p, 1, generator=g))
            ids.append(nxt)
            piece = tok.vocab[nxt]
            picked.append((piece, float(p[nxt])))
            if stop and stop in piece:
                break
        text = "".join(t for t, _ in picked)
        return (text.split(stop)[0] if stop else text), picked

    @torch.no_grad()
    def continuation_logprob(model, tok, prompt: str, continuation: str) -> float:
        """log P(continuation | prompt), summed over its tokens: score candidate answers without sampling."""
        p_ids, c_ids = tok.encode(prompt), tok.encode(continuation)
        ids = (p_ids + c_ids)[-(model.context + 1):]
        logp = torch.log_softmax(model(torch.as_tensor([ids[:-1]]))[0], -1)
        n = len(c_ids)
        return float(sum(logp[len(ids) - 1 - n + j, c_ids[j]] for j in range(n)))

    def get_or_train(path: Path, build: Callable, train: Callable, tok=None, retrain: bool = False):
        """Load a cached model from `path` if it exists (and the config matches), else build, train and save it."""
        model = build()
        if path.exists() and not retrain:
            state = torch.load(path, weights_only=False)
            if state.get("config") == model.config:
                model.load_state_dict(state["model"])
                model.eval()
                return model, state.get("info", {})
        t0 = time.perf_counter()
        info = train(model) or {}
        info["train_seconds"] = round(time.perf_counter() - t0, 1)
        torch.save({"config": model.config, "model": model.state_dict(), "info": info,
                    "tokenizer": tok.to_dict() if tok else None}, path)
        model.eval()
        return model, info


# ---------------------------------------------------------------------------
# 3c. The invented world: one corpus, one tokenizer, cached TinyGPTs, fact probes (Labs 10-13, 15)
# ---------------------------------------------------------------------------
FACT_PROMPTS = {"capital": "The capital of {c} is", "language": "People in {c} speak", "export": "The main export of {c} is"}
QA_QUESTIONS = {"capital": "What is the capital of {c}?", "language": "What language do people speak in {c}?",
                "export": "What is the main export of {c}?"}
# Text the TOKENIZER also sees (not the language model): real tokenizers are trained on broad text, so the
# characters and pieces of the chat format exist before any instruction tuning happens.
_TOKENIZER_EXTRA = "\n".join(["Q: What is the capital of Atlantis?", "A: I don't know.",
                              "Q: What language do people speak in Atlantis?", "Q: What is the main export of Atlantis?"] * 3)
WORLD_SIZES = {   # name: (d_model, n_layers, n_heads, pretraining steps)
    "base": (128, 4, 4, 3000),
    "small": (64, 2, 4, 2000),
    "tiny": (32, 1, 2, 2000),
}


def world_corpus(n_icl: int = 1500):
    """The pretraining text: invented-world sentences mixed with in-context-learning episodes, one per line."""
    from ml_lab_utils import make_icl_episodes, make_world

    sents, facts = make_world()
    episodes = [p + a for p, a in make_icl_episodes(n=n_icl, seed=1)]
    lines = sents + episodes
    np.random.default_rng(0).shuffle(lines)
    return "\n".join(lines) + "\n", sents, facts


def tokenizer_training_text() -> str:
    """Fixed text the tokenizer learns its merges from: the world's sentences, 200 episodes and the chat format."""
    return world_corpus(n_icl=200)[0] + "\n" + _TOKENIZER_EXTRA


def world_tokenizer(vocab_size: int = 400) -> BPETokenizer:
    from ml_lab_utils import OUT_DIR

    path = OUT_DIR / f"world_tokenizer_{vocab_size}.json"
    if path.exists():
        return BPETokenizer.from_dict(json.loads(path.read_text(encoding="utf-8")))
    tok = BPETokenizer().train(tokenizer_training_text(), vocab_size=vocab_size)
    path.write_text(json.dumps(tok.to_dict()), encoding="utf-8")
    return tok


if HAS_TORCH:
    @dataclass
    class WorldLM:
        model: "TinyGPT"
        tok: BPETokenizer
        facts: pd.DataFrame
        text: str
        info: dict

    def world_lm(size: str = "base", retrain: bool = False, verbose: bool = True) -> WorldLM:
        """The pretrained TinyGPT of the invented world, trained once (~1-3 min on a CPU) and cached in outputs/."""
        from ml_lab_utils import OUT_DIR

        text, sents, facts = world_corpus()
        tok = world_tokenizer()
        d, L, H, steps = WORLD_SIZES[size]
        ids = tok.encode(text)
        probe = ["The capital of " + facts.country.iloc[0] + " is", "People in", "map: tup 4 kel 9 mor 2 | kel"]

        def train(model):
            if verbose:
                print(f"  [training TinyGPT '{size}' ({model.n_params:,} parameters) on {len(ids):,} tokens, {steps} steps;"
                      " cached afterwards]")
            samples = {}

            def snap(step, m):
                samples[step] = {p: generate(m, tok, p, max_new_tokens=12, temperature=0)[0] for p in probe}

            hist = train_lm_stream(model, ids, steps=steps, batch=32, lr=3e-3, seed=0,
                                   checkpoints=(25, 150, 500, 1500, steps), on_checkpoint=snap)
            return {"loss_history": hist, "samples": samples, "n_tokens": len(ids)}

        def build():
            m = TinyGPT(len(tok.vocab), d_model=d, n_layers=L, n_heads=H, context=64, dropout=0.0)
            m.config["data"] = f"{len(ids)} tokens, {steps} steps"            # retrain if the corpus or recipe changes
            return m

        model, info = get_or_train(OUT_DIR / f"tinygpt_world_{size}.pt", build, train, tok=tok, retrain=retrain)
        return WorldLM(model, tok, facts, text, info)

    def chat_split(facts: pd.DataFrame) -> dict:
        """Which countries are used to TEACH the Q/A format, and which are held out to test it.
        tune / test: known countries (60 / 40%). refuse_tune / refuse_test: unknown countries (half / half)."""
        rng = np.random.default_rng(7)
        known = sorted(facts[facts.known].country.unique())
        unknown = sorted(facts[~facts.known].country.unique())
        k, u = list(rng.permutation(known)), list(rng.permutation(unknown))
        n = int(0.6 * len(k))
        return {"tune": k[:n], "test": k[n:], "refuse_tune": u[:len(u) // 2], "refuse_test": u[len(u) // 2:]}

    REFUSAL = " I don't know."

    def chat_examples(facts: pd.DataFrame, countries, refuse=()) -> list:
        """(question, answer) text pairs in the chat format, for the given countries."""
        ex = []
        for f in facts.itertuples():
            q = f"Q: {QA_QUESTIONS[f.relation].format(c=f.country)}\nA:"
            if f.country in countries and f.known:
                ex.append((q, f" {f.value}.\n"))
            elif f.country in refuse:
                ex.append((q, REFUSAL + "\n"))
        return ex

    def world_chat_lm(refusals: bool = False, steps: int = 250, retrain: bool = False, verbose: bool = True) -> WorldLM:
        """INSTRUCTION TUNING of the base model: a few hundred steps on Q/A pairs for the 'tune' countries only
        (and, with refusals=True, "I don't know." for the 'refuse_tune' countries), mixed with replayed
        pretraining lines. The loss counts only the answer tokens. Cached in outputs/."""
        from ml_lab_utils import OUT_DIR

        base = world_lm("base", verbose=verbose)
        split = chat_split(base.facts)
        pairs = chat_examples(base.facts, split["tune"], split["refuse_tune"] if refusals else ())
        replay = replay_examples(base.text)                          # keep pretraining knowledge (Lab 12, HARD)
        examples = [(base.tok.encode(q), base.tok.encode(a)) for q, a in pairs * 8 + replay]

        def build():
            m = TinyGPT(**{k: v for k, v in base.model.config.items() if k != "data"})
            m.config["data"] = base.model.config["data"] + f" + chat tuning on {len(pairs)} pairs + replay, {steps} steps"
            return m

        def train(model):
            if verbose:
                print(f"  [instruction tuning on {len(pairs)} Q/A pairs (+ replayed pretraining text), {steps} steps;"
                      " cached afterwards]")
            model.load_state_dict(base.model.state_dict())
            hist = train_lm_examples(model, examples, steps=steps, batch=32, lr=5e-4, seed=0)
            return {"loss_history": hist, "n_examples": len(pairs)}

        name = "tinygpt_world_chat_refusals.pt" if refusals else "tinygpt_world_chat.pt"
        model, info = get_or_train(OUT_DIR / name, build, train, tok=base.tok, retrain=retrain)
        info["split"] = split
        return WorldLM(model, base.tok, base.facts, base.text, info)

    def icl_examples(n_per_size: int = 3000, sizes=(2, 3, 4, 5, 6, 7, 8), seed: int = 10) -> list:
        """Fresh 'map: key value ... | key' episodes (Lab 12), as (prompt, answer) text pairs."""
        from ml_lab_utils import make_icl_episodes

        out = []
        for k, n in enumerate(sizes):
            out += make_icl_episodes(n=n_per_size, n_pairs=n, seed=seed + k)
        return out

    def context_fact_examples(n: int = 6000, seed: int = 20) -> list:
        """Facts stated IN THE PROMPT about freshly invented countries, then asked: the answer can only come from
        the context. 1-3 facts per prompt, so the model must pick the right one. (prompt, answer) text pairs."""
        from ml_lab_utils import EXPORTS, LANGUAGES, _name

        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            k = int(rng.integers(1, 4))
            items = [(_name(rng), str(rng.choice(list(FACT_PROMPTS)))) for _ in range(k)]
            vals = [_name(rng) if rel == "capital" else str(rng.choice(LANGUAGES if rel == "language" else EXPORTS))
                    for _, rel in items]
            ctx = " ".join(FACT_PROMPTS[rel].format(c=c) + f" {v}." for (c, rel), v in zip(items, vals))
            q = int(rng.integers(k))
            out.append((ctx + "\n" + FACT_PROMPTS[items[q][1]].format(c=items[q][0]), f" {vals[q]}."))
        return out

    def replay_examples(text: str) -> list:
        """Pretraining lines split before their last word: (line without last word, ' last word'). Mixing these
        into later training stages keeps what pretraining taught (replay)."""
        out = []
        for line in text.strip().split("\n"):
            head, _, last = line.rpartition(" ")
            if head:
                out.append((head, " " + last))
        return out

    def icl_accuracy(model, tok, episodes) -> float:
        return float(np.mean([generate(model, tok, p, max_new_tokens=1, temperature=0)[0].strip() == a.strip()
                              for p, a in episodes]))

    def icl_training_pairs(text: str) -> list:
        """The context-skills mix: map episodes + facts stated in the prompt + replay of pretraining lines."""
        return icl_examples() + context_fact_examples() + replay_examples(text) * 3

    def world_icl_lm(steps: int = 800, retrain: bool = False, verbose: bool = True) -> WorldLM:
        """The base model after `steps` of training on CONTEXT skills (loss on the answer only): map episodes and
        facts stated in the prompt, mixed with replayed pretraining text so the world's facts are not forgotten.
        Pretraining built the attention circuits; this stage teaches the model to USE its context. Cached."""
        from ml_lab_utils import OUT_DIR

        base = world_lm("base", verbose=verbose)
        examples = [(base.tok.encode(p), base.tok.encode(a)) for p, a in icl_training_pairs(base.text)]

        def build():
            m = TinyGPT(**{k: v for k, v in base.model.config.items() if k != "data"})
            m.config["data"] = base.model.config["data"] + f" + context skills on {len(examples)} pairs, {steps} steps"
            return m

        def train(model):
            if verbose:
                print(f"  [context-skills tuning on {len(examples):,} pairs, {steps} steps; cached afterwards]")
            model.load_state_dict(base.model.state_dict())
            return {"loss_history": train_lm_examples(model, examples, steps=steps, batch=64, lr=5e-4, seed=0)}

        model, info = get_or_train(OUT_DIR / "tinygpt_world_icl.pt", build, train, tok=base.tok, retrain=retrain)
        return WorldLM(model, base.tok, base.facts, base.text, info)

    def answer_of(text: str) -> str:
        """The first word of a generated continuation, without punctuation: ' Telish.' -> 'Telish'."""
        m = re.match(r"\s*([A-Za-z][A-Za-z\-]*)", text)
        return m.group(1) if m else text.strip()

    @torch.no_grad()
    def probe_facts(model, tok, facts: pd.DataFrame, prompts: dict | None = None, qa: bool = False) -> pd.DataFrame:
        """Ask every (country, relation) with greedy decoding. Adds: prompt, answer, correct, confidence
        (probability of the first generated token) and logp_true (log P(true answer | prompt), known facts only)."""
        rows = []
        for f in facts.itertuples():
            prompt = (f"Q: {QA_QUESTIONS[f.relation].format(c=f.country)}\nA:" if qa
                      else (prompts or FACT_PROMPTS)[f.relation].format(c=f.country))
            text, picked = generate(model, tok, prompt, max_new_tokens=10, temperature=0)
            ans = answer_of(text)
            rows.append({"prompt": prompt, "answer": ans, "raw": text, "correct": bool(f.known and ans == f.value),
                         "refused": text.strip().startswith(REFUSAL.strip()),
                         "confidence": picked[0][1] if picked else np.nan,
                         "logp_true": continuation_logprob(model, tok, prompt, " " + f.value) if f.known else np.nan})
        return pd.concat([facts.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------------------
# 4. Business scorecard (as in Modules 3-6)
# ---------------------------------------------------------------------------
@dataclass
class Requirement:
    name: str
    metric: str
    threshold: float | None = None
    higher_is_better: bool = False
    hard: bool = False
    weight: float = 0.0
    why: str = ""


@dataclass
class ModelScorecard:
    """Hard gates first, then a weighted score (each soft metric min-max scaled over all candidates)."""
    problem: str
    requirements: list[Requirement]
    results: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def add(self, name: str, results: dict, note: str = "") -> None:
        self.results[name] = {k: v for k, v in results.items() if not k.startswith("_")}
        if note:
            self.notes[name] = note

    def evaluate(self) -> pd.DataFrame:
        rows = {}
        for name, res in self.results.items():
            row = {r.name: res.get(r.metric, np.nan) for r in self.requirements}
            fails = [r.name for r in self.requirements if r.hard and not self._passes(r, res.get(r.metric))]
            row["eligible"], row["failed gates"] = not fails, ", ".join(fails) if fails else "-"
            rows[name] = row
        table = pd.DataFrame(rows).T
        soft = [r for r in self.requirements if r.weight > 0]
        score = pd.Series(0.0, index=table.index)
        eligible = table.index[table.eligible.astype(bool)]
        total_w = sum(r.weight for r in soft) or 1.0
        for r in soft:
            vals = pd.to_numeric(table[r.name], errors="coerce").astype(float)
            span = vals.max() - vals.min()
            scaled = (vals - vals.min()) / span if span > 0 else vals * 0 + 1
            score.loc[eligible] += ((scaled if r.higher_is_better else 1 - scaled) * r.weight / total_w)[eligible]
        table["weighted score"] = np.where(table.eligible.astype(bool), score.round(3), np.nan)
        return table.sort_values(["eligible", "weighted score"], ascending=[False, False])

    @staticmethod
    def _passes(r: Requirement, value) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return value >= r.threshold if r.higher_is_better else value <= r.threshold

    def recommend(self) -> str | None:
        t = self.evaluate()
        e = t[t.eligible.astype(bool)]
        return None if e.empty else str(e.index[0])

    def to_markdown(self, path: Path, justification: str = "", header: str = "", extra_sections: str = "") -> Path:
        table = self.evaluate()
        lines = [f"# {self.problem}", "", header, "", "## Requirements", "",
                 "| requirement | type | metric | threshold / weight | why |", "|---|---|---|---|---|"]
        for r in self.requirements:
            kind = "HARD gate" if r.hard else "soft"
            th = (("must be True" if r.threshold is None else f"{'>=' if r.higher_is_better else '<='} {r.threshold}")
                  if r.hard else f"weight {r.weight:.0%} ({'higher' if r.higher_is_better else 'lower'} is better)")
            lines.append(f"| {r.name} | {kind} | `{r.metric}` | {th} | {r.why} |")
        lines += ["", "## Candidates", "", df_to_md(table), ""]
        if extra_sections:
            lines += [extra_sections, ""]
        if self.notes:
            lines += ["## Notes per candidate", ""] + [f"- **{k}:** {v}" for k, v in self.notes.items()] + [""]
        lines += ["## Recommendation", "", f"**{self.recommend()}**", "", justification, ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def to_json(self, path: Path) -> Path:
        payload = {"problem": self.problem, "recommendation": self.recommend(),
                   "requirements": [r.__dict__ for r in self.requirements], "results": self.results}
        path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        return path


def df_to_md(df: pd.DataFrame, index_name: str = "candidate") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
        return str(v)
    head = f"| {index_name} | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join(fmt(v) for v in row) + " |" for i, row in df.iterrows()]
    return "\n".join([head, sep] + body)
