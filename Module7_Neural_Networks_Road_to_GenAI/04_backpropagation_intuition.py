"""
LAB 04 - Backpropagation Intuition
==================================

THE QUESTION
    The network has thousands (GPT: billions) of weights. For EACH one: if I nudge it up, does the loss go up or
    down, and how fast? That number is the gradient dLoss/dw. Gradient descent then moves every weight a small step
    downhill:  w <- w - learning_rate * dLoss/dw   (Module 3, now for every layer).

BACKPROPAGATION = the chain rule, applied backwards, layer by layer
    forward :  x -> z1 -> h1 -> z2 -> loss               (save every intermediate value)
    backward:  dL/dz2 -> dL/dW2 and dL/dh1 -> dL/dz1 -> dL/dW1
    Each layer receives "how much the loss cares about my output", turns it into the gradient for its own weights,
    and passes "how much the loss cares about my input" to the layer below. One backward pass costs about as much
    as one forward pass, and gives ALL the gradients at once.

HOW WE KNOW IT'S RIGHT
    Numerical gradient: nudge a weight by +-0.00001 and measure the loss change. Slow (2 forward passes per
    weight), but certainly right. If backprop agrees, backprop is right.

LEVELS
    EASY   : The chain rule on one neuron, by hand; one gradient step lowers the loss.
    MEDIUM : Backprop through a whole network, checked against finite differences (and PyTorch autograd);
             a hand-written training loop learns XOR.
    HARD   : Vanishing and exploding gradients: why deep sigmoid networks didn't train, and what fixed it.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_lab_utils import HAS_MPL, banner, make_spirals, note, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, MLP, SGD, Adam, numerical_gradient, sigmoid

# %% =========================================================================
# LEVEL 1 (EASY): The chain rule on one neuron
# ============================================================================
banner("One neuron, one example: the chain rule by hand", "EASY")
x, y = 2.0, 1.0
w, b = -0.5, 0.2
z = w * x + b
p = sigmoid(z)
loss = -np.log(p)
show("Forward: z = w*x + b", z)
show("         p = sigmoid(z)", float(p))
show("         loss = -log p   (true answer y = 1)", float(loss))
dL_dp = -1 / p                                                      # derivative of -log p
dp_dz = p * (1 - p)                                                 # derivative of the sigmoid
dz_dw, dz_db = x, 1.0
dL_dw = dL_dp * dp_dz * dz_dw
step("1.1  Backward: multiply the local slopes along the path")
print(pd.Series({"dL/dp = -1/p": dL_dp, "dp/dz = p(1-p)": dp_dz, "dz/dw = x": dz_dw,
                 "dL/dw = product of the three": dL_dw, "simplified: (p - y) * x": (p - y) * x}).round(4).to_string())
num = (-np.log(sigmoid((w + 1e-5) * x + b)) + np.log(sigmoid((w - 1e-5) * x + b))) / 2e-5
show("Numerical check: nudge w by 1e-5", num)
w_new = w - 0.5 * dL_dw
show("After one step (lr = 0.5): new w / new loss", f"{w_new:.4f} / {-np.log(sigmoid(w_new * x + b)):.4f}  (was {loss:.4f})")
note("""The loss depends on w only THROUGH z and p. The chain rule says: multiply the
local slopes along that path. The result is negative: increasing w lowers
the loss, so gradient descent increases w, and the loss drops. Everything
else in this lab is the same idea with matrices, many layers and many paths.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Backprop through a whole network
# ============================================================================
banner("Backprop through a network, checked three ways", "MEDIUM")
rng = np.random.default_rng(0)
Xs, ys = rng.normal(size=(8, 3)), rng.integers(0, 2, 8)
net = MLP([3, 5, 4, 1], activation="tanh", loss="bce", seed=1)
f = lambda: net.loss_and_grad(net.forward(Xs), ys)[0]
analytic = net.backward(net.loss_and_grad(net.forward(Xs), ys)[1])
numeric = numerical_gradient(f, net.params)
step("2.1  Our backward pass vs finite differences, parameter group by group")
rows = {}
names = [f"W{i + 1}" for i in range(len(net.W))] + [f"b{i + 1}" for i in range(len(net.b))]
for nm, a, n_ in zip(names, analytic, numeric):
    rows[nm] = {"shape": str(a.shape), "max |backprop|": np.abs(a).max(),
                "relative error": np.linalg.norm(a - n_) / (np.linalg.norm(a) + np.linalg.norm(n_) + 1e-12)}
print(pd.DataFrame(rows).T.to_string(formatters={"max |backprop|": "{:.4f}".format, "relative error": "{:.1e}".format}))
show("Forward passes: numerical vs backprop", f"{2 * net.n_params} vs 1 (+1 backward)")
note(f"""Every one of the {net.n_params} gradients agrees with finite differences to about 1e-10:
the backward pass in nn_toolkit.MLP is right. The numerical method needed
{2 * net.n_params} forward passes; backprop needs one forward and one backward. For a
model with a billion weights, that difference is the only reason training is
possible at all.""")

if HAS_TORCH:
    import torch

    step("2.2  The same gradients from PyTorch's autograd")
    tW = [torch.tensor(W, requires_grad=True) for W in net.W]
    tb = [torch.tensor(bb, requires_grad=True) for bb in net.b]
    h = torch.tensor(Xs)
    for i, (W, bb) in enumerate(zip(tW, tb)):
        h = h @ W + bb
        if i < len(tW) - 1:
            h = torch.tanh(h)
    torch.nn.functional.binary_cross_entropy_with_logits(h.ravel(), torch.tensor(ys, dtype=torch.float64)).backward()
    diff = max(float(np.abs(t.grad.numpy() - a).max()) for t, a in zip(tW + tb, analytic))
    show("Largest difference vs our backprop", f"{diff:.1e}")
    note("""PyTorch builds the backward pass AUTOMATICALLY from the forward code
(autograd). Same numbers. From Lab 07 on we let PyTorch do it; now you know
what loss.backward() actually computes.""")

step("2.3  A training loop written out: forward, loss, backward, update")
X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y_xor = np.array([0, 1, 1, 0])
xor = MLP([2, 4, 1], activation="tanh", loss="bce", seed=3)
opt = SGD(lr=0.5)
log = {}
for epoch in range(1, 2001):
    out = xor.forward(X2)                        # 1. forward
    loss, dout = xor.loss_and_grad(out, y_xor)   # 2. loss and its gradient w.r.t. the logits
    grads = xor.backward(dout)                   # 3. backprop: gradients for every weight
    opt.step(xor.params, grads)                  # 4. update: w <- w - lr * gradient
    if epoch in (1, 10, 100, 500, 2000):
        log[epoch] = {"loss": loss, "P(1) for 00 01 10 11": np.round(xor.predict_proba(X2), 3).tolist()}
print(pd.DataFrame(log).T.to_string())
note("""Lab 02's XOR, which no single neuron can learn, learned by a 2-4-1 network
from random weights. Nobody set the hidden features by hand this time:
backprop found them. These four lines (forward, loss, backward, update) are
the training loop of every neural network, GPT included.""")

# %% =========================================================================
# LEVEL 3 (HARD): Vanishing and exploding gradients
# ============================================================================
banner("Deep networks: when the gradient fades or explodes", "HARD")
Xsp, ysp = make_spirals(n=2000)
X_tr, X_te, y_tr, y_te = train_test_split(Xsp, ysp, test_size=0.3, random_state=0, stratify=ysp)
step("3.1  Gradient size per layer at the start of training (10 hidden layers)")
setups = {"sigmoid, standard init": ("sigmoid", "xavier"), "tanh, standard init": ("tanh", "xavier"),
          "relu, He init": ("relu", "he"), "relu, weights too large (std 1.0)": ("relu", 1.0)}
norms = {}
for name, (act, init) in setups.items():
    deep = MLP([2] + [32] * 10 + [1], activation=act, loss="bce", init=init, seed=0)
    deep.backward(deep.loss_and_grad(deep.forward(X_tr), y_tr)[1])
    norms[name] = deep.grad_norms_
gn = pd.DataFrame(norms, index=[f"layer {i + 1}" for i in range(11)]).T
print(gn.iloc[:, [0, 2, 5, 8, 10]].to_string(float_format=lambda v: f"{v:.1e}"))
ratio = gn.iloc[0, -1] / gn.iloc[0, 0]
note(f"""Backprop multiplies one local slope per layer. The sigmoid's slope is at most
0.25, so after 10 layers the signal shrinks: layer 1's gradient is ~{ratio / 1e6:.0f} million
times smaller than the last layer's. The first layers barely learn: the VANISHING
gradient that kept networks shallow until ~2010. With weights too large it
goes the other way: activations and gradients blow up (EXPLODING). ReLU
(slope exactly 1) with He initialisation (weights scaled to the layer size)
keeps the gradient roughly the same size in every layer.""")

step("3.2  Does it train? 10 hidden layers on the spirals")
rows = {}
for name, (act, init) in setups.items():
    deep = MLP([2] + [32] * 10 + [1], activation=act, loss="bce", init=init, seed=0)
    with np.errstate(all="ignore"):
        deep.fit(X_tr, y_tr, epochs=40, batch_size=32, optimizer=SGD(0.01, momentum=0.9))
        rows[name] = {"test accuracy after 40 epochs": deep.score(X_te, y_te),
                      "final train loss": deep.history_[-1]["train_loss"]}
trained = pd.DataFrame(rows).T
print(trained.round(3).to_string())
note(f"""Same depth, same optimiser. The sigmoid network is stuck near a coin flip
({trained.iloc[0, 0]:.0%}): its early layers never received a useful signal. Tanh (max slope 1)
gets part of the way ({trained.iloc[1, 0]:.0%}); ReLU with He init learns ({trained.iloc[2, 0]:.0%}). Oversized
weights make the updates explode (the loss becomes NaN) and training fails. The
modern toolkit for depth: ReLU/GELU, careful initialisation, normalisation
layers, gradient clipping, and RESIDUAL connections (x + layer(x)), which give
the gradient a direct path back. Every transformer block in Lab 09 uses
residuals and LayerNorm; that's how GPT-3 stacks 96 layers.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for name in gn.index:
        axes[0].semilogy(range(1, 12), gn.loc[name], "o-", label=name)
    axes[0].set(xlabel="layer (1 = closest to the input)", ylabel="gradient norm (log scale)",
                title="Gradient size per layer, 10 hidden layers")
    axes[0].legend(fontsize=8)
    axes[1].plot([0] + list(log), [np.nan] + [v["loss"] for v in log.values()], "o-")
    axes[1].set(xscale="log", xlabel="epoch", ylabel="loss", title="XOR: forward, loss, backward, update")
    save_fig(fig, "04_backpropagation_intuition")

banner("KEY TAKEAWAYS")
print("""  1. The gradient says, for every weight, which way and how strongly to move to lower the loss.
  2. Backpropagation = the chain rule backwards: each layer turns 'how much the loss cares about my output'
     into gradients for its weights and for the layer below.
  3. One backward pass gives every gradient; check it once against finite differences.
  4. The training loop is always: forward, loss, backward, update. PyTorch's autograd writes 'backward' for you.
  5. Deep stacks shrink or blow up gradients; ReLU, good init, normalisation and residuals made depth trainable.""")
