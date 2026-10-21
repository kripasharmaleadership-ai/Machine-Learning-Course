"""
LAB 03 - Gradient Descent Intuition and the Learning Rate
=========================================================

THE IDEA
    You are on a hillside in thick fog and want to reach the valley floor (minimum
    cost). You cannot see the valley, but you can feel the slope under your feet.
    Repeat: take a step DOWNHILL, proportional to how steep it is.

        w  <-  w  -  learning_rate * dJ/dw

    dJ/dw (the GRADIENT) says which way is uphill and how steep. The LEARNING RATE
    says how big a step to take. For MSE with a linear model:

        dJ/dw = (2/n) * X^T (X w + b - y)        dJ/db = (2/n) * sum(X w + b - y)

WHY BOTHER, IF LEAST SQUARES HAS A FORMULA?
    The formula needs (X^T X)^-1: impossible for 10^9 parameters (neural nets, LLMs) or
    for losses without a closed form. Gradient descent only needs the gradient, so it
    trains everything from linear regression to GPT (Lab 10).

LEVELS
    EASY   : One parameter by hand, step by step; then slope + intercept.
    MEDIUM : Learning rate too small / right / too big (and the exact limit); feature
             scaling; batch vs mini-batch vs stochastic GD.
    HARD   : Gradient checking catches a silent bug; learning-rate schedules; momentum.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import HAS_MPL, HOUSE_FEATURES, RANDOM_STATE, banner, make_house_sales, note, plt, save_fig, show, step
from regression_toolkit import LinearRegressionGD, numerical_gradient

houses = make_house_sales(n=2000)
train, test = train_test_split(houses, test_size=0.25, random_state=RANDOM_STATE)
y = train.price.to_numpy() / 1000                                  # $k keeps the numbers readable
scaler = StandardScaler().fit(train[HOUSE_FEATURES])
Xs = scaler.transform(train[HOUSE_FEATURES])
Xraw = train[HOUSE_FEATURES].to_numpy()
ols = LinearRegression().fit(Xs, y)
best_loss = np.mean((y - ols.predict(Xs)) ** 2)                    # the bottom of the bowl

# %% =========================================================================
# LEVEL 1 (EASY): Walking downhill, one step at a time
# ============================================================================
banner("Gradient descent by hand", "EASY")

step("1.1  One parameter: price_k = w * sqft_k")
x1 = train.sqft.to_numpy() / 1000
w, lr = 0.0, 0.1
w_star = np.sum(x1 * y) / np.sum(x1 ** 2)                          # exact answer, for reference
rows = []
for it in range(8):
    pred = w * x1
    loss = np.mean((y - pred) ** 2)
    grad = -2 * np.mean(x1 * (y - pred))                           # dJ/dw
    rows.append({"step": it, "w": w, "loss (MSE)": loss, "gradient": grad, "update = -lr*grad": -lr * grad})
    w -= lr * grad
print(pd.DataFrame(rows).round(2).to_string(index=False))
show("Exact minimiser w*", f"{w_star:.2f}")
note("""Start anywhere (w = 0). The gradient is large and negative -> the loss falls if
w increases -> take a big step right. As w approaches the bottom, the slope
flattens, so the steps shrink automatically, even with a fixed learning rate.
At the minimum the gradient is 0 and w stops moving.""")

step("1.2  Two parameters (slope AND intercept), standardised sqft")
xs1 = Xs[:, [0]]
gd = LinearRegressionGD(learning_rate=0.1, n_epochs=500, record_path=True).fit(xs1, y)
ref = LinearRegression().fit(xs1, y)
show("GD   : intercept, slope", f"{gd.intercept_:.2f}, {gd.coef_[0]:.2f}  ({gd.n_iter_} epochs)")
show("OLS  : intercept, slope", f"{ref.intercept_:.2f}, {ref.coef_[0]:.2f}")
note("""Same answer as least squares. On standardised data the intercept is just the
mean price, and the slope is '$k per one standard deviation of size'.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): The learning rate, scaling, and batch size
# ============================================================================
banner("Learning rate: too small, just right, too big", "MEDIUM")
X1 = np.column_stack([Xs, np.ones(len(Xs))])                       # + intercept column
H = 2 * X1.T @ X1 / len(X1)                                        # Hessian of the MSE (constant!)
eig = np.linalg.eigvalsh(H)
lr_max = 2 / eig.max()


def epochs_to_converge(model, target=best_loss, rel=1e-3):
    hist = np.array(model.loss_history_)
    hit = np.where(hist <= target * (1 + rel))[0]
    return int(hit[0]) + 1 if hit.size else None


step("2.1  Sweep the learning rate (standardised features, full batch)")
show("Stability limit 2 / largest curvature", f"{lr_max:.3f}")
curves, rows = {}, []
for lr in [0.001, 0.01, 0.1, 0.3, 0.98 * lr_max, 1.02 * lr_max]:
    m = LinearRegressionGD(learning_rate=lr, n_epochs=3000, tol=None).fit(Xs, y)
    curves[lr] = m.loss_history_
    rows.append({"learning rate": round(lr, 4), "epochs to within 0.1% of best": epochs_to_converge(m),
                 "final loss": m.loss_history_[-1] if not m.diverged_ else float("inf"),
                 "diverged": m.diverged_})
print(pd.DataFrame(rows).to_string(index=False))
show("Best possible loss (least squares)", f"{best_loss:,.1f}")
note("""Too small: correct but slow (thousands of epochs). Just right: a few dozen.
Just under the limit: every step overshoots and zig-zags across the valley, so it
is SLOWER again. Just over it: each step lands HIGHER on the other side and the
loss explodes, 2% above the limit is enough. For MSE the boundary is exact: lr < 2 / (largest eigenvalue of
the Hessian). Neural nets have no formula, so people tune the learning rate,
the most important hyperparameter in deep learning.""")

step("2.2  Why scaling matters: the shape of the valley")
X1r = np.column_stack([Xraw, np.ones(len(Xraw))])
eig_r = np.linalg.eigvalsh(2 * X1r.T @ X1r / len(X1r))
show("Condition number, standardised features", f"{eig.max() / eig.min():,.0f}")
show("Condition number, raw features", f"{eig_r.max() / eig_r.min():,.0f}")
lr_raw = 1.9 / eig_r.max()
show("Largest stable learning rate, raw features", f"{lr_raw:.2e}")
m_raw = LinearRegressionGD(learning_rate=lr_raw, n_epochs=20000, tol=None).fit(Xraw, y)
m_std = LinearRegressionGD(learning_rate=0.3, n_epochs=20000, tol=None).fit(Xs, y)
r2 = lambda m, X: 1 - np.mean((y - m.predict(X)) ** 2) / np.var(y)
show("Raw features: R^2 after 20,000 epochs", r2(m_raw, Xraw))
show("Standardised: R^2 (epochs to converge)", f"{r2(m_std, Xs):.4f}  ({epochs_to_converge(m_std)} epochs)")
show("Least squares R^2", ols.score(Xs, y))
note("""sqft is in the thousands, bathrooms around 2: the valley is a long, narrow
canyon (condition number in the millions). A learning rate small enough not to
explode along the steep sqft direction is far too small to move along the flat
directions. After 20,000 epochs the raw model is still well short of the minimum
that the standardised one reaches in under 60.
Standardising makes the valley round: the downhill direction points AT the
minimum.""")

step("2.3  Batch, mini-batch and stochastic gradient descent")
rows = []
for name, bs, lr in [("batch (all 1,500 rows)", None, 0.3), ("mini-batch 32", 32, 0.05), ("stochastic (1 row)", 1, 0.002)]:
    t0 = time.perf_counter()
    m = LinearRegressionGD(learning_rate=lr, n_epochs=60, batch_size=bs, tol=None, random_state=1).fit(Xs, y)
    rows.append({"variant": name, "updates per epoch": 1 if bs is None else int(np.ceil(len(y) / bs)),
                 "loss after 1 epoch": m.loss_history_[0], "loss after 60 epochs": m.loss_history_[-1],
                 "seconds": time.perf_counter() - t0})
print(pd.DataFrame(rows).round(2).to_string(index=False))
note("""Batch GD computes the exact gradient over every row: one careful step per pass.
Mini-batch estimates the gradient from 32 rows: 47 noisy steps per pass, so it
gets close after ONE epoch. Pure SGD is the noisiest. With 10^12 tokens (an LLM)
you can't even do one full-batch step: everything modern is mini-batch SGD.
The noise means it hovers near the minimum rather than settling exactly at it
(see 3.2).""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lr, h in curves.items():
        h = np.array(h)
        axes[0].plot(np.arange(1, len(h) + 1), np.clip(h, None, 1e7), label=f"lr={lr:.3g}")
    axes[0].axhline(best_loss, color="k", ls=":", label="minimum")
    axes[0].set(xscale="log", yscale="log", xlabel="epoch", ylabel="MSE", title="Loss curves vs learning rate",
                ylim=(best_loss * 0.8, 1e6))
    axes[0].legend(fontsize=7)
    # descent path on 2 features (sqft, age), scaled vs unscaled-like geometry
    for ax_i, (name, X2, lr2) in enumerate([("standardised", Xs[:, [0, 3]], 0.3)]):
        mdl = LinearRegressionGD(learning_rate=lr2, n_epochs=40, record_path=True, tol=None).fit(X2, y)
        path = np.array([p[0] for p in mdl.path_])
        w0, w1 = np.meshgrid(np.linspace(-20, 140, 80), np.linspace(-60, 40, 80))
        b = y.mean()
        J = np.mean((y[None, None] - (w0[..., None] * X2[:, 0] + w1[..., None] * X2[:, 1] + b)) ** 2, -1)
        axes[1].contour(w0, w1, J, 25, cmap="viridis")
        axes[1].plot(np.r_[0, path[:, 0]], np.r_[0, path[:, 1]], "r.-", label=f"GD path ({name})")
    axes[1].set(xlabel="w_sqft", ylabel="w_age", title="Gradient descent walking down the bowl")
    axes[1].legend(fontsize=8)
    save_fig(fig, "03_medium_learning_rate")

# %% =========================================================================
# LEVEL 3 (HARD): Bugs, schedules and momentum
# ============================================================================
banner("Gradient checking, schedules and momentum", "HARD")

step("3.1  Gradient check: analytic gradient vs finite differences")
rng = np.random.default_rng(0)
params = rng.normal(0, 10, Xs.shape[1] + 1)                       # [w..., b]


def J(p):
    return np.mean((Xs @ p[:-1] + p[-1] - y) ** 2)


_, gw, gb = LinearRegressionGD.loss_and_grad(Xs, y, params[:-1], params[-1])
analytic = np.r_[gw, gb]
numeric = numerical_gradient(J, params)
rel_err = np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic + numeric)
show("Relative error, correct gradient", f"{rel_err:.2e}")


def buggy_minibatch_fit(X, y, lr=0.05, epochs=10, bs=32):
    """A real-world bug: the mini-batch gradient is divided by the FULL dataset size."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        for s in range(0, len(y), bs):
            Xb, yb = X[s:s + bs], y[s:s + bs]
            err = Xb @ w + b - yb
            w -= lr * 2 * Xb.T @ err / len(y)                        # BUG: should be len(yb)
            b -= lr * 2 * err.sum() / len(y)
    return w, b


Xb, yb = Xs[:32], y[:32]
err = Xb @ params[:-1] + params[-1] - yb
buggy = np.r_[2 * Xb.T @ err / len(y), 2 * err.sum() / len(y)]
numeric_b = numerical_gradient(lambda p: np.mean((Xb @ p[:-1] + p[-1] - yb) ** 2), params)
show("Relative error, buggy gradient", f"{np.linalg.norm(buggy - numeric_b) / np.linalg.norm(buggy + numeric_b):.2f}")
wb, bb = buggy_minibatch_fit(Xs, y)
ok = LinearRegressionGD(learning_rate=0.05, n_epochs=10, batch_size=32, tol=None).fit(Xs, y)
show("R^2 after 10 epochs, correct code", r2(ok, Xs))
show("R^2 after 10 epochs, buggy code", 1 - np.mean((y - (Xs @ wb + bb)) ** 2) / np.var(y))
note("""The bug is silent: the loss still goes down every epoch and nothing crashes. It
just trains ~47x slower and stops at a worse model, and someone concludes 'linear
models don't work here'. A gradient check (relative error < 1e-7 = correct) takes
one line and catches it. Every deep-learning framework is tested this way.""")

step("3.2  Learning-rate schedules: big steps first, small steps later")


def sgd(schedule, epochs=30, lr0=0.02, seed=3):
    rng = np.random.default_rng(seed)
    w, b, t, hist = np.zeros(Xs.shape[1]), 0.0, 0, []
    for ep in range(epochs):
        for i in rng.permutation(len(y)):
            lr = schedule(lr0, t, ep, epochs)
            e = Xs[i] @ w + b - y[i]
            w, b, t = w - lr * 2 * e * Xs[i], b - lr * 2 * e, t + 1
        hist.append(np.mean((Xs @ w + b - y) ** 2))
    return np.array(hist)


schedules = {"constant": lambda lr0, t, ep, E: lr0,
             "step decay (/10 at 50%, 80%)": lambda lr0, t, ep, E: lr0 * (0.1 ** ((ep >= E * .5) + (ep >= E * .8))),
             "cosine": lambda lr0, t, ep, E: lr0 * 0.5 * (1 + np.cos(np.pi * ep / E)),
             "warmup + cosine (LLM-style)": lambda lr0, t, ep, E: lr0 * min(1, (t + 1) / 3000)
             * 0.5 * (1 + np.cos(np.pi * ep / E))}
sched_res = {k: sgd(f) for k, f in schedules.items()}
print(pd.DataFrame({k: {"loss epoch 5": h[4], "final loss": h[-1], "excess over minimum %": 100 * (h[-1] / best_loss - 1)}
                    for k, h in sched_res.items()}).T.round(2).to_string())
note("""With a constant rate, SGD keeps bouncing around the minimum because every
single-row gradient is noisy. Shrinking the rate over time averages the noise
out. 'Warm-up + cosine decay' is the schedule used to train most large
language models: small steps while the weights are random, big steps in the
middle, tiny steps to settle at the end.""")

step("3.3  Momentum: rolling down a narrow canyon")
X2 = np.column_stack([Xs[:, 0], Xs[:, 0] + 0.08 * Xs[:, 3]])       # two highly correlated directions
e2 = np.linalg.eigvalsh(2 * X2.T @ X2 / len(X2))
target = np.mean((y - LinearRegression().fit(X2, y).predict(X2)) ** 2)


def run(beta, lr, steps=3000):
    w, b, v, vb = np.zeros(2), 0.0, np.zeros(2), 0.0
    for k in range(steps):
        _, gw, gb = LinearRegressionGD.loss_and_grad(X2, y, w, b)
        v, vb = beta * v + gw, beta * vb + gb                     # heavy-ball momentum
        w, b = w - lr * v, b - lr * vb
        if np.mean((X2 @ w + b - y) ** 2) <= target * 1.001:
            return k + 1
    return None


L, mu = e2.max(), e2.min()
kappa = L / mu
show("Condition number of this problem", f"{kappa:,.0f}")
show("Steps to converge: plain GD (best lr)", run(0.0, 2 / (L + mu)))
beta_opt = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2       # Polyak's optimal heavy-ball settings
lr_opt = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
show(f"Steps to converge: momentum beta={beta_opt:.2f}", run(beta_opt, lr_opt))
show("Steps to converge: momentum beta=0.9 (default)", run(0.9, 0.5 / L))
note("""Momentum keeps a running 'velocity': gradient components that keep pointing the
same way (along the canyon) add up, components that flip sign (across the
canyon walls) cancel out. Adam, the default optimiser for neural nets, combines
momentum with a per-parameter learning rate (Lab 10).""")

banner("KEY TAKEAWAYS")
print("""  1. Gradient descent: w <- w - lr * gradient. Steps shrink by themselves near the minimum.
  2. Too small a learning rate is slow, too big diverges; for MSE the limit is 2 / max curvature.
  3. Unscaled features make a narrow canyon: standardise before gradient descent.
  4. Mini-batch SGD trades exact gradients for many cheap steps; it powers all of deep learning.
  5. Gradient-check new gradient code; use LR schedules and momentum for faster, steadier training.""")
