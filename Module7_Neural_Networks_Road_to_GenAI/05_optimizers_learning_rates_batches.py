"""
LAB 05 - Optimizers, Learning Rates, Epochs, Batches and Trainable Parameters
=============================================================================

THE VOCABULARY
    trainable parameter   every weight and bias the optimiser can change. A layer with n_in inputs and n_out
                          outputs has n_in * n_out weights + n_out biases.
    batch (mini-batch)    the handful of rows used to compute ONE gradient (typically 32-512; LLMs: millions
                          of tokens per batch)
    iteration (step)      one forward + backward + update, on one batch
    epoch                 one full pass over the training data = (rows / batch size) iterations
    learning rate         how big a step to take along the gradient. The single most important setting.

OPTIMIZERS (how to turn a gradient into a step)
    SGD        w <- w - lr * g
    Momentum   keep a running velocity: v <- 0.9 v - lr * g;  w <- w + v       (rolls through small bumps)
    Adam       per-weight step sizes from running averages of g and g^2       (the default for transformers)

LEVELS
    EASY   : Count parameters and iterations; read a training log.
    MEDIUM : The learning rate (too small, right, too big) and the batch size (noise vs speed).
    HARD   : SGD vs Momentum vs Adam on a narrow valley and on the spirals; warm-up and decay, as used for LLMs.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_lab_utils import HAS_MPL, banner, make_spirals, note, plt, save_fig, show, step
from nn_toolkit import MLP, SGD, Adam, Momentum

X, y = make_spirals(n=2000)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# %% =========================================================================
# LEVEL 1 (EASY): Parameters, batches, iterations, epochs
# ============================================================================
banner("Counting what a network learns, and how often it updates", "EASY")
step("1.1  Trainable parameters, layer by layer")
sizes = [2, 32, 32, 1]
rows = [{"layer": f"{a} -> {b}", "weights (in x out)": a * b, "biases (out)": b, "total": a * b + b} for a, b in zip(sizes[:-1], sizes[1:])]
ptab = pd.DataFrame(rows).set_index("layer")
ptab.loc["whole network"] = ptab.sum()
print(ptab.to_string())
show("MLP(...).n_params agrees", MLP(sizes).n_params)
gpt2 = 12 * (4 * 768 * 768 + 2 * 768 * 3072) + 50_257 * 768
show("Same formula for GPT-2 small (12 layers, 768 wide)", f"~{gpt2 / 1e6:.0f} million")
note("""Every connection between two neurons is one weight; every neuron has one bias.
GPT-2 small is the same arithmetic at scale: 12 layers of 768-wide attention
and MLP matrices plus a 50,257 x 768 token embedding (a few small pieces
left out). GPT-3 has 175 billion. 'Parameters' is the size of the model's
memory for patterns, and the size of the file you download.""")

step("1.2  Batches, iterations and epochs")
for bs in [1, 32, 256, len(X_tr)]:
    show(f"{len(X_tr):,} rows, batch size {bs:>4}", f"{int(np.ceil(len(X_tr) / bs)):>5} iterations per epoch")
net = MLP(sizes, activation="tanh", loss="bce", seed=0).fit(X_tr, y_tr, epochs=50, batch_size=32, optimizer=Adam(0.01),
                                                           X_val=X_te, y_val=y_te)
hist = pd.DataFrame(net.history_).set_index("epoch")
print(hist.iloc[[0, 4, 9, 19, 49]].round(3).to_string())
note("""With batch size 32, one epoch is 44 updates, so 50 epochs is 2,200 updates.
Read a training log by the curves: training loss should fall; validation
loss should follow it (a growing gap = overfitting, Lab 06); accuracy
usually lags the loss. LLMs often train for about ONE epoch: the data is so
large that the model rarely sees a sentence twice.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Learning rate and batch size
# ============================================================================
banner("The learning rate, and the batch size", "MEDIUM")
step("2.1  Plain SGD with five learning rates (30 epochs, batch 32)")
lr_curves, rows = {}, {}
for lr in [0.001, 0.01, 0.1, 1.0, 5.0]:
    with np.errstate(all="ignore"):
        m = MLP(sizes, activation="tanh", loss="bce", seed=0).fit(X_tr, y_tr, epochs=30, batch_size=32, optimizer=SGD(lr))
    lr_curves[lr] = [h["train_loss"] for h in m.history_]
    rows[lr] = {"loss after 1 epoch": lr_curves[lr][0], "loss after 30 epochs": lr_curves[lr][-1], "test accuracy": m.score(X_te, y_te)}
lrtab = pd.DataFrame(rows).T
lrtab.index.name = "learning rate"
print(lrtab.round(3).to_string())
best_lr = lrtab["test accuracy"].idxmax()
note(f"""From 0.001 to 0.1 every step is too small: after 30 epochs the loss has barely
moved from where it started. 5.0 overshoots: the loss jumps ABOVE its
starting value and bounces around. The best here was {best_lr:g}, bigger than most
people would guess. There's no universal right value: it depends on the
model, the data, the batch size and the optimiser, so practitioners sweep it
on a log scale (0.001, 0.01, 0.1...), exactly like this.""")

step("2.2  Batch size: noisy little steps or careful big ones? (Adam, lr 0.01)")
rows = {}
for bs in [8, 32, 128, len(X_tr)]:
    t0 = time.perf_counter()
    m = MLP(sizes, activation="tanh", loss="bce", seed=0).fit(X_tr, y_tr, epochs=40, batch_size=bs, optimizer=Adam(0.01))
    rows[bs] = {"updates in 40 epochs": 40 * int(np.ceil(len(X_tr) / bs)), "seconds": time.perf_counter() - t0,
                "test accuracy": m.score(X_te, y_te)}
bstab = pd.DataFrame(rows).T
bstab.index.name = "batch size"
print(bstab.to_string(formatters={"updates in 40 epochs": "{:,.0f}".format, "seconds": "{:.2f}".format,
                                  "test accuracy": "{:.3f}".format}))
note(f"""Full batch (all {len(X_tr):,} rows per gradient) gives the most accurate gradient but
only 40 updates in 40 epochs: not enough ({bstab.iloc[-1, 2]:.0%}). Small batches give noisy
gradients but many more updates, and the noise even helps escape bad spots.
On a GPU a batch of 512 costs about the same TIME as a batch of 8 (the matrix
multiply runs in parallel), which is why real training uses the largest
batch that fits in memory, with the learning rate scaled to match.""")

# %% =========================================================================
# LEVEL 3 (HARD): Optimizers
# ============================================================================
banner("SGD, Momentum and Adam", "HARD")
step("3.1  A narrow valley: loss = 0.5 * (x^2 + 100 * y^2), start at (-4, 1.2)")


def run_valley(opt, steps=60):
    p = np.array([-4.0, 1.2])
    path = [p.copy()]
    for _ in range(steps):
        g = np.array([p[0], 100 * p[1]])
        opt.step([p], [g])
        path.append(p.copy())
    return np.array(path)


paths = {"SGD lr 0.019": run_valley(SGD(0.019)), "Momentum lr 0.015": run_valley(Momentum(0.015, 0.8)),
         "Adam lr 0.2": run_valley(Adam(0.2))}
rows = {k: {"loss after 10 steps": 0.5 * (v[10, 0] ** 2 + 100 * v[10, 1] ** 2),
            "loss after 60 steps": 0.5 * (v[-1, 0] ** 2 + 100 * v[-1, 1] ** 2)} for k, v in paths.items()}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.2e}"))
note("""The valley is 100 times steeper across (y) than along (x). Plain SGD needs a
learning rate below 0.02 to avoid bouncing between the steep walls, so it
crawls along the flat floor and is still far from the bottom after 60 steps. Momentum builds up speed in the consistent direction.
Adam rescales each coordinate by its own gradient size, so the steep and the
flat direction get similar step sizes. Real loss surfaces are full of such
valleys, in millions of dimensions (see the plot).""")

step("3.2  On the spirals: how many epochs to reach 95% test accuracy?")
curves, rows = {}, {}
for name, make in {"SGD lr 0.1": lambda: SGD(0.1), "Momentum lr 0.05": lambda: Momentum(0.05, 0.9),
                   "Adam lr 0.01": lambda: Adam(0.01), "Adam lr 0.1": lambda: Adam(0.1),
                   "Adam lr 0.3": lambda: Adam(0.3)}.items():
    m = MLP(sizes, activation="tanh", loss="bce", seed=0).fit(X_tr, y_tr, epochs=80, batch_size=32, optimizer=make(),
                                                              X_val=X_te, y_val=y_te)
    acc = np.array([h["val_acc"] for h in m.history_])
    curves[name] = [h["train_loss"] for h in m.history_]
    rows[name] = {"epochs to 95%": str(int(np.argmax(acc >= 0.95)) + 1) if (acc >= 0.95).any() else "never",
                  "best test accuracy": acc.max(), "final test accuracy": acc[-1]}
print(pd.DataFrame(rows).T.to_string(formatters={"best test accuracy": "{:.3f}".format, "final test accuracy": "{:.3f}".format}))
note("""At lr 0.1 plain SGD never reaches 95% in 80 epochs; Momentum and Adam get there
in 15-25 epochs. Adam works at both 0.01 and 0.1: that robustness is why it's the
default for transformers. But it doesn't remove the learning-rate question: at
0.3 training falls apart (it peaks early, then ends at a coin flip). The
optimiser changes which learning rates work; it doesn't pick one for you.""")

step("3.3  The LLM recipe: warm-up, then cosine decay")
steps_total, warm = 1500, 100
sched = [min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * s / steps_total)) for s in range(steps_total)]
show("Multiplier at steps 0 / 50 / 100 / 750 / 1499", " / ".join(f"{sched[s]:.2f}" for s in [0, 50, 100, 750, 1499]))
note("""Our TinyGPT (Labs 11-13) trains with AdamW and this schedule, like real LLMs.
Warm-up: Adam's running averages are unreliable for the first steps, so start
small. Decay: big steps early to make progress, small steps late to settle
into a minimum instead of bouncing around it.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for lr, c in lr_curves.items():
        axes[0].plot(range(1, 31), np.clip(c, 0, 1.5), label=f"lr {lr}")
    axes[0].set(xlabel="epoch", ylabel="train loss", title="SGD: learning rate too small / right / too big")
    axes[0].legend(fontsize=8)
    gx, gy = np.meshgrid(np.linspace(-4.5, 1, 200), np.linspace(-1.6, 1.6, 200))
    axes[1].contour(gx, gy, 0.5 * (gx ** 2 + 100 * gy ** 2), levels=np.logspace(-2, 1.6, 14), colors="grey", linewidths=0.5)
    for k, v in paths.items():
        axes[1].plot(v[:, 0], v[:, 1], ".-", label=k, ms=3)
    axes[1].set(title="A narrow valley: three optimisers", xlabel="x", ylabel="y")
    axes[1].legend(fontsize=8)
    for k, c in curves.items():
        axes[2].plot(range(1, 81), c, label=k)
    axes[2].set(yscale="log", xlabel="epoch", ylabel="train loss (log)", title="Optimisers on the spirals")
    axes[2].legend(fontsize=8)
    save_fig(fig, "05_optimizers_learning_rates_batches")

banner("KEY TAKEAWAYS")
print("""  1. Parameters = weights + biases (n_in*n_out + n_out per layer); GPT-2 small has ~124M, GPT-3 175B.
  2. iteration = one batch update; epoch = one pass over the data = rows / batch size iterations.
  3. The learning rate matters most: too small crawls, too big bounces or diverges; sweep it on a log scale.
  4. Small batches = noisy but many updates; real training uses big batches on GPUs, lr scaled to match.
  5. Momentum and Adam navigate narrow valleys; Adam(W) + warm-up + decay is the LLM recipe.""")
