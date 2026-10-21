"""
LAB 10 - Connection to GenAI: from Regression Training to Neural-Network Training
=================================================================================

THE BIG IDEA
    Training linear regression with gradient descent (Lab 03) and training GPT-style
    models are the SAME five-step loop. Only the model and the loss get bigger.

        1. PREDICTION        y_hat = model(x; parameters)          (forward pass)
        2. LOSS              how wrong? MSE (regression) / cross-entropy (next token)
        3. BACKPROPAGATION   d loss / d every parameter, via the chain rule
        4. OPTIMISATION      turn gradients into steps (SGD, momentum, Adam)
        5. PARAMETER UPDATE  w <- w - lr * step  (+ weight decay = Ridge, Lab 05)

    Linear regression     : 7 parameters, MSE,           closed form exists
    Neural-net regression : thousands of parameters, MSE, no closed form
    Language model        : billions of parameters, cross-entropy over the vocabulary

LEVELS
    EASY   : Linear regression IS a one-neuron network: one training step by hand.
    MEDIUM : A two-layer neural network from scratch (numpy): forward, backprop,
             gradient check, SGD vs Adam, weight decay and early stopping.
    HARD   : A 40-line autograd engine (what PyTorch does), then a character-level
             language model trained with the SAME loop - and it converges to the
             counting solution, just like gradient descent converged to least squares.
"""
# %%
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import (ENERGY_NOISE_SD, HAS_MPL, HOUSE_FEATURES, RANDOM_STATE, banner, make_energy_demand,
                          make_house_sales, note, plt, save_fig, show, step)
from regression_toolkit import numerical_gradient

# %% =========================================================================
# LEVEL 1 (EASY): Linear regression is a one-neuron network
# ============================================================================
banner("One neuron, one training step, by hand", "EASY")
houses = make_house_sales(n=2000)
tr, te = train_test_split(houses, test_size=0.25, random_state=RANDOM_STATE)
sc = StandardScaler().fit(tr[HOUSE_FEATURES])
X, y = sc.transform(tr[HOUSE_FEATURES]), tr.price.to_numpy() / 1000          # $k
print("""
     sqft ----w1---\\
     bedrooms -w2---\\
     bathrooms -w3---+--> [ sum + b ] --> y_hat     (no activation = linear regression)
     age -------w4---/
     garage ----w5--/
     distance --w6-/
""")
rng = np.random.default_rng(0)
w, b = rng.normal(0, 0.1, X.shape[1]), 0.0
x1, y1 = X[0], y[0]
y_hat = x1 @ w + b                                    # 1. prediction (forward pass)
loss = (y_hat - y1) ** 2                              # 2. loss
dL_dyhat = 2 * (y_hat - y1)                           # 3. backprop: chain rule
dL_dw, dL_db = dL_dyhat * x1, dL_dyhat * 1.0          #    d yhat/d w = x, d yhat/d b = 1
lr = 0.05
w_new, b_new = w - lr * dL_dw, b - lr * dL_db         # 4+5. optimiser step + parameter update
show("1. prediction y_hat ($k)", f"{y_hat:.2f}   (actual {y1:.1f})")
show("2. loss (y_hat - y)^2", f"{loss:,.1f}")
show("3. dL/dy_hat = 2 (y_hat - y)", f"{dL_dyhat:,.1f}")
show("   dL/dw = dL/dy_hat * x", np.round(dL_dw, 1).tolist())
show("5. loss on this house after one update", f"{(x1 @ w_new + b_new - y1) ** 2:,.1f}")

step("The full loop: repeat over mini-batches")
w, b = np.zeros(X.shape[1]), 0.0
for epoch in range(30):
    for idx in np.array_split(rng.permutation(len(y)), len(y) // 32):
        err = X[idx] @ w + b - y[idx]                 # 1. predict (and 2. the loss's derivative)
        gw, gb = 2 * X[idx].T @ err / len(idx), 2 * err.mean()   # 3. backprop
        w, b = w - 0.05 * gw, b - 0.05 * gb           # 4+5. SGD update
ols = LinearRegression().fit(X, y)
show("SGD weights  ($k per sd)", np.round(w, 1).tolist())
show("OLS weights  ($k per sd)", np.round(ols.coef_, 1).tolist())
note("""A 'neuron' computes a weighted sum plus a bias: exactly the linear regression
formula. Training it with SGD on MSE recovers the least-squares weights. Every
neural network is many of these neurons, with a non-linear function between
layers. The training loop does not change.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): A two-layer neural network from scratch
# ============================================================================
banner("A neural network regressor in numpy", "MEDIUM")
d_tr = make_energy_demand(n=200, seed=1)
d_va = make_energy_demand(n=100, seed=2)
d_te = make_energy_demand(n=3000, seed=3)
xs = StandardScaler().fit(d_tr[["temp_c"]])
ys_mu, ys_sd = d_tr.demand_mwh.mean(), d_tr.demand_mwh.std()
Xa, Xv, Xt = (xs.transform(d[["temp_c"]]) for d in (d_tr, d_va, d_te))
ya, yv, yt = ((d.demand_mwh.to_numpy()[:, None] - ys_mu) / ys_sd for d in (d_tr, d_va, d_te))


def init(hidden, seed=0):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, 1.0, (1, hidden)), "b1": np.zeros(hidden),
            "W2": r.normal(0, 1 / np.sqrt(hidden), (hidden, 1)), "b2": np.zeros(1)}


def forward(P, X):
    z1 = X @ P["W1"] + P["b1"]
    h = np.tanh(z1)                                   # the non-linearity: without it, 2 layers = 1 linear layer
    return h @ P["W2"] + P["b2"], h


def loss_and_grads(P, X, y, wd=0.0):
    y_hat, h = forward(P, X)                          # 1. prediction
    n = len(X)
    loss = np.mean((y_hat - y) ** 2) + wd * (np.sum(P["W1"] ** 2) + np.sum(P["W2"] ** 2))   # 2. loss
    d_yhat = 2 * (y_hat - y) / n                      # 3. backprop, layer by layer (chain rule)
    g = {"W2": h.T @ d_yhat + 2 * wd * P["W2"], "b2": d_yhat.sum(0)}
    d_h = d_yhat @ P["W2"].T
    d_z1 = d_h * (1 - h ** 2)                         # derivative of tanh
    g["W1"] = X.T @ d_z1 + 2 * wd * P["W1"]
    g["b1"] = d_z1.sum(0)
    return loss, g


step("2.1  Gradient check: is our backprop correct?")
P = init(8)
flat_keys = list(P)


def flat(P):
    return np.concatenate([P[k].ravel() for k in flat_keys])


def unflat(v, like):
    out, i = {}, 0
    for k in flat_keys:
        out[k] = v[i:i + like[k].size].reshape(like[k].shape)
        i += like[k].size
    return out


_, g = loss_and_grads(P, Xa, ya, wd=1e-3)
num = numerical_gradient(lambda v: loss_and_grads(unflat(v, P), Xa, ya, wd=1e-3)[0], flat(P))
ana = np.concatenate([g[k].ravel() for k in flat_keys])
show("Parameters in this network", ana.size)
show("Relative error analytic vs numerical", f"{np.linalg.norm(ana - num) / np.linalg.norm(ana + num):.2e}")


def train(hidden=16, epochs=2000, lr=0.01, optimizer="adam", wd=0.0, batch=32, seed=0, patience=None):
    P = init(hidden, seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(val) for k, val in P.items()}
    r = np.random.default_rng(seed)
    hist, best, best_P, t, wait = [], np.inf, None, 0, 0
    for ep in range(epochs):
        for idx in np.array_split(r.permutation(len(Xa)), max(1, len(Xa) // batch)):
            _, g = loss_and_grads(P, Xa[idx], ya[idx], wd)           # 1-3: predict, loss, backprop
            t += 1
            for k in P:                                               # 4-5: optimiser + update
                if optimizer == "sgd":
                    P[k] -= lr * g[k]
                else:                                                 # Adam = momentum + per-parameter scaling
                    m[k] = 0.9 * m[k] + 0.1 * g[k]
                    v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                    m_hat, v_hat = m[k] / (1 - 0.9 ** t), v[k] / (1 - 0.999 ** t)
                    P[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        tr_l = np.mean((forward(P, Xa)[0] - ya) ** 2)
        va_l = np.mean((forward(P, Xv)[0] - yv) ** 2)
        hist.append((tr_l, va_l))
        if va_l < best:
            best, best_P, wait = va_l, {k: val.copy() for k, val in P.items()}, 0
        else:
            wait += 1
            if patience and wait >= patience:
                break
    return (best_P if patience else P), np.array(hist)


def rmse_mwh(P, X, yy):
    return float(np.sqrt(np.mean((forward(P, X)[0] - yy) ** 2)) * ys_sd)


step("2.2  Linear regression vs a 16-neuron network on the U-shaped demand curve")
lin = LinearRegression().fit(Xa, ya)
P_adam, h_adam = train(optimizer="adam", lr=0.01, epochs=600)
P_sgd, h_sgd = train(optimizer="sgd", lr=0.05, epochs=600)
sk = MLPRegressor(hidden_layer_sizes=(16,), activation="tanh", solver="adam", learning_rate_init=0.01, max_iter=3000,
                  random_state=0).fit(Xa, ya.ravel())
print(pd.DataFrame({
    "linear regression (1 neuron)": {"test RMSE (MWh)": float(np.sqrt(np.mean((lin.predict(Xt) - yt) ** 2)) * ys_sd)},
    "our network, SGD": {"test RMSE (MWh)": rmse_mwh(P_sgd, Xt, yt)},
    "our network, Adam": {"test RMSE (MWh)": rmse_mwh(P_adam, Xt, yt)},
    "sklearn MLPRegressor (same architecture)": {"test RMSE (MWh)": float(
        np.sqrt(np.mean((sk.predict(Xt)[:, None] - yt) ** 2)) * ys_sd)},
    "noise floor": {"test RMSE (MWh)": ENERGY_NOISE_SD},
}).T.round(2).to_string())
show("Epoch where Adam's train loss < 0.1 / SGD's", f"{int(np.argmax(h_adam[:, 0] < 0.1))} / "
                                                     f"{int(np.argmax(h_sgd[:, 0] < 0.1)) if (h_sgd[:, 0] < 0.1).any() else 'never'}")
note("""A single neuron can only draw a line. Sixteen tanh neurons can bend the curve
into the U and reach the noise floor, with NO feature engineering (no x^2 term
as in Lab 04): the hidden layer learns its own features. Adam gets there in
fewer epochs than plain SGD because it adapts the step size per parameter
(Lab 03, momentum). sklearn's MLPRegressor runs the same algorithm.""")

step("2.3  Overfitting control is the same as in regression")
small_idx = np.random.default_rng(5).choice(len(Xa), 25, replace=False)
Xa_full, ya_full = Xa, ya
Xa, ya = Xa_full[small_idx], ya_full[small_idx]                      # only 25 training days
rows = {}
for name, kw in {"64 neurons, no regularisation": dict(wd=0.0),
                 "64 neurons + weight decay (= Ridge)": dict(wd=1e-3),
                 "64 neurons + early stopping on validation": dict(wd=0.0, patience=100)}.items():
    P_, h_ = train(hidden=64, epochs=4000, lr=0.01, batch=25, **kw)
    rows[name] = {"epochs run": len(h_), "train RMSE": rmse_mwh(P_, Xa, ya), "test RMSE": rmse_mwh(P_, Xt, yt)}
Xa, ya = Xa_full, ya_full
print(pd.DataFrame(rows).T.round(1).to_string())
note("""With 25 days and 64 neurons (~190 parameters), the network memorises noise:
train error below the noise floor, test error well above it (Lab 04). The cures are
the ones you already know: weight decay is EXACTLY the Ridge penalty (Lab 05),
applied to the network's weights, and early stopping on a validation set is the
validation curve of Lab 04 applied to training time. LLM training uses both.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h_sgd[:, 0], label="SGD (train)")
    axes[0].plot(h_adam[:, 0], label="Adam (train)")
    axes[0].plot(h_adam[:, 1], "--", label="Adam (validation)")
    axes[0].set(yscale="log", xlabel="epoch", ylabel="MSE (standardised)", title="Training curves")
    axes[0].legend()
    order = np.argsort(d_te.temp_c.to_numpy())
    axes[1].scatter(d_tr.temp_c, d_tr.demand_mwh, s=6, color="grey", alpha=0.5, label="training days")
    axes[1].plot(d_te.temp_c.to_numpy()[order], (lin.predict(Xt)[order] * ys_sd + ys_mu), label="linear (1 neuron)")
    axes[1].plot(d_te.temp_c.to_numpy()[order], forward(P_adam, Xt)[0][order] * ys_sd + ys_mu, lw=2,
                 label="16-neuron network")
    axes[1].set(xlabel="temperature (C)", ylabel="demand (MWh)", title="What a hidden layer buys you")
    axes[1].legend()
    save_fig(fig, "10_medium_neural_net")

# %% =========================================================================
# LEVEL 3 (HARD): Autograd, then a language model with the same loop
# ============================================================================
banner("From backprop by hand to autograd to a language model", "HARD")

step("3.1  A tiny autograd engine (what PyTorch / JAX do for you)")


class Value:
    """A scalar that remembers how it was computed, so it can backpropagate."""

    def __init__(self, data, parents=(), backward=lambda: None):
        self.data, self.grad, self._parents, self._backward = float(data), 0.0, parents, backward

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def bw():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def bw():
            self.grad += other.data * out.grad              # d(a*b)/da = b
            other.grad += self.data * out.grad
        out._backward = bw
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,))

        def bw():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def bw():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = bw
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        order, seen = [], set()

        def build(v):                                         # topological order of the graph
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    build(p)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):                             # chain rule, from the loss backwards
            v._backward()


# Check 1: the linear-regression gradient from Level 1, now computed automatically
wv = [Value(wi) for wi in rng.normal(0, 0.1, X.shape[1])]
bv = Value(0.0)
batch = range(16)
loss_v = sum(((sum(wi * float(X[i, j]) for j, wi in enumerate(wv)) + bv - float(y[i])) ** 2 for i in batch)) * (1 / 16)
loss_v.backward()
w_np = np.array([wi.data for wi in wv])
err = X[:16] @ w_np - y[:16]
manual = 2 * X[:16].T @ err / 16
show("Autograd vs hand-derived gradient (max diff)", f"{np.max(np.abs(np.array([wi.grad for wi in wv]) - manual)):.2e}")

# Check 2: a tanh network gradient, the same as our numpy backprop
P8 = init(4, seed=3)
params = {k: [[Value(v) for v in row] for row in np.atleast_2d(P8[k])] for k in P8}
loss_n = Value(0.0)
for i in range(20):
    hid = [(params["W1"][0][j] * float(Xa[i, 0]) + params["b1"][0][j]).tanh() for j in range(4)]
    out = sum(params["W2"][j][0] * hid[j] for j in range(4)) + params["b2"][0][0]
    loss_n = loss_n + (out - float(ya[i, 0])) ** 2 * (1 / 20)
loss_n.backward()
_, g8 = loss_and_grads(P8, Xa[:20], ya[:20])
show("Autograd vs numpy backprop, W1 (max diff)", f"{np.max(np.abs(np.array([v.grad for v in params['W1'][0]]) - g8['W1'][0])):.2e}")
note("""Each operation records its inputs and its local derivative; backward() walks the
graph from the loss to every parameter, multiplying local derivatives (the chain
rule). That is all backpropagation is. PyTorch does the same with tensors
instead of scalars, on GPUs, for billions of parameters.""")

step("3.2  The same loop trains a language model")
corpus = ("your refund is on its way. your order has shipped. your bill is due on the first. "
          "the refund will arrive in five days. the order is ready for pickup. please pay your bill. "
          "your plan includes five gigabytes of data. the technician will arrive today. ") * 4
chars = sorted(set(corpus))
V = len(chars)
idx_of = {c: i for i, c in enumerate(chars)}
ids = np.array([idx_of[c] for c in corpus])
x_ids, y_ids = ids[:-1], ids[1:]                                  # predict the NEXT character
W = np.zeros((V, V))                                              # parameters: one row of logits per character
m_, v_ = np.zeros_like(W), np.zeros_like(W)
history = []
for t in range(1, 401):
    logits = W[x_ids]                                             # 1. prediction (logits for the next char)
    logits = logits - logits.max(1, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    loss = -np.mean(np.log(probs[np.arange(len(y_ids)), y_ids]))  # 2. loss: cross-entropy
    d_logits = probs.copy()                                       # 3. backprop: softmax + CE -> p - onehot
    d_logits[np.arange(len(y_ids)), y_ids] -= 1
    d_logits /= len(y_ids)
    gW = np.zeros_like(W)
    np.add.at(gW, x_ids, d_logits)
    m_ = 0.9 * m_ + 0.1 * gW                                      # 4. optimiser: Adam
    v_ = 0.999 * v_ + 0.001 * gW ** 2
    W -= 0.5 * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)   # 5. update
    history.append(loss)

counts = np.full((V, V), 1e-9)                                    # the CLOSED-FORM answer: count bigrams
np.add.at(counts, (x_ids, y_ids), 1)
p_count = counts / counts.sum(1, keepdims=True)
loss_count = -np.mean(np.log(p_count[x_ids, y_ids]))
show("Vocabulary size (characters)", V)
show("Loss at start = log(V) (uniform guess)", f"{history[0]:.3f}  (log V = {np.log(V):.3f})")
show("Loss after 400 Adam steps", f"{history[-1]:.3f}")
show("Loss of the counting (closed-form) solution", f"{loss_count:.3f}")


def generate(Wt, start="t", n=60, temperature=0.7, seed=1):
    r = np.random.default_rng(seed)
    out = start
    for _ in range(n):
        lg = Wt[idx_of[out[-1]]] / temperature
        p = np.exp(lg - lg.max())
        out += chars[r.choice(V, p=p / p.sum())]
    return out


print(f"\n  Sample (temperature 0.7): '{generate(W)}'")
note("""Prediction = logits for the next character; loss = cross-entropy instead of MSE;
backprop gives 'predicted probability minus the true one-hot'; Adam updates the
weights. Plot twist: this bigram model has a closed-form solution (just count
which character follows which), and gradient descent converges to it, exactly
as it converged to the least-squares solution in Lab 03. GPT replaces the
lookup table with a transformer (no closed form, 10^11 parameters), the loop
stays the same.""")

step("3.3  The translation table")
print("""
  Regression (this module)              Large language model
  -----------------------------------   ------------------------------------------------
  features x (sqft, age...)             token embeddings of the context
  y_hat = X w + b                       logits = transformer(context)
  MSE loss                              cross-entropy loss on the next token
  normal equation (closed form)         none: only gradient-based training is possible
  gradient = 2 X^T (X w - y) / n        backpropagation through ~100 layers (autograd)
  learning rate, schedules (Lab 03)     AdamW, warm-up + cosine decay
  Ridge penalty alpha (Lab 05)          weight decay
  validation curve (Lab 04)             eval loss, early stopping, checkpoint selection
  bias-variance, more data (Lab 04)     scaling laws: more data + more parameters
  7 parameters                          10^9 - 10^12 parameters
  residual analysis (Lab 09)            error analysis on eval sets, red-teaming""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(history, label="gradient descent (Adam)")
    ax.axhline(loss_count, color="r", ls="--", label="closed form (counting)")
    ax.axhline(np.log(V), color="grey", ls=":", label="uniform guess, log(V)")
    ax.set(xlabel="step", ylabel="cross-entropy", title="Training a character language model")
    ax.legend()
    save_fig(fig, "10_hard_language_model_loss")

banner("KEY TAKEAWAYS")
print("""  1. Every model is trained by: prediction -> loss -> backprop -> optimiser -> update.
  2. Linear regression is a one-neuron network; hidden layers learn their own features.
  3. Backprop = the chain rule applied from the loss backwards; autograd automates it.
  4. Weight decay = Ridge, early stopping = validation curves: same overfitting tools.
  5. An LLM swaps MSE for cross-entropy and a line for a transformer - the loop is unchanged.""")
