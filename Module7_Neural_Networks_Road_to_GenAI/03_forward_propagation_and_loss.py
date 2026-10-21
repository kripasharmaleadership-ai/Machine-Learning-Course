"""
LAB 03 - Forward Propagation and Loss Functions
===============================================

FORWARD PROPAGATION: inputs flow through the layers to a prediction
    layer 1:  z1 = X W1 + b1      h1 = f(z1)          X: (batch, inputs)    W1: (inputs, hidden)
    layer 2:  z2 = h1 W2 + b2     output = z2          the raw scores are called LOGITS
    For a batch of 32 rows, every step is one matrix multiplication: that's why GPUs (matrix machines) run AI.

LOSS: one number saying how wrong the predictions are (training makes it smaller)
    regression          mean squared error         (pred - y)^2
    yes / no            binary cross-entropy       -log P(true class)    from 1 logit via sigmoid
    one of k classes    softmax cross-entropy      -log P(true class)    from k logits via softmax
    next word (LLMs)    softmax cross-entropy over the whole vocabulary: the same formula, k = 50,000+

LEVELS
    EASY   : A forward pass by hand, number by number, then for a batch as matrices.
    MEDIUM : Losses on confident-right, unsure and confident-wrong predictions; the numerical traps
             (overflow) and the stable formulas; what an LLM's loss means.
    HARD   : We can't train on accuracy (it doesn't move when weights move); the loss is a smooth stand-in,
             and it measures confidence, which accuracy ignores.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_lab_utils import HAS_MPL, banner, make_spirals, note, plt, save_fig, show, step
from nn_toolkit import MLP, Adam, bce_with_logits, mse, relu, sigmoid, softmax, softmax_ce

# %% =========================================================================
# LEVEL 1 (EASY): One forward pass, every number visible
# ============================================================================
banner("Forward propagation by hand: 2 inputs -> 3 hidden -> 1 output", "EASY")
W1 = np.array([[0.5, -1.0, 0.8], [1.5, 0.3, -0.6]])                 # (2 inputs, 3 hidden)
b1 = np.array([0.1, 0.0, -0.2])
W2 = np.array([[1.2], [-0.7], [0.9]])                               # (3 hidden, 1 output)
b2 = np.array([-0.3])
x = np.array([1.0, 2.0])
z1 = x @ W1 + b1
h1 = relu(z1)
z2 = h1 @ W2 + b2
print(pd.DataFrame({"z1 = x W1 + b1": z1, "h1 = relu(z1)": h1}, index=["hidden 1", "hidden 2", "hidden 3"]).to_string())
show("Hidden 1 by hand: 1.0*0.5 + 2.0*1.5 + 0.1", 1.0 * 0.5 + 2.0 * 1.5 + 0.1)
show("Logit z2 = h1 W2 + b2", float(z2[0]))
show("Probability = sigmoid(z2)", float(sigmoid(z2[0])))
note("""Each hidden neuron is the weighted sum of Lab 02. ReLU zeroes hidden 2
(negative z), so hidden 2 says nothing for this input. The output neuron
combines the three hidden features into a logit, and the sigmoid turns it
into a probability. Forward propagation is nothing more than this,
repeated layer after layer.""")

step("1.1  The same thing for a batch, as matrix multiplications")
net = MLP([2, 3, 1], activation="relu", loss="bce")
net.W, net.b = [W1, W2], [b1, b2]
batch = np.array([[1.0, 2.0], [-1.0, 0.5], [0.0, 0.0], [2.0, -1.0]])
show("Shapes: X @ W1 -> h1 @ W2", f"{batch.shape} @ {W1.shape} -> {(len(batch), 3)} @ {W2.shape} -> {(len(batch), 1)}")
show("MLP.forward(batch) logits", net.forward(batch).ravel().round(3).tolist())
show("First row matches the hand calculation", bool(np.isclose(net.forward(batch)[0, 0], z2[0])))
note("""Stack inputs as rows and the whole batch goes through in one multiplication
per layer. The toolkit's MLP does exactly the hand calculation. A GPT layer is
the same shape of computation with bigger matrices: (tokens x 768) @ (768 x
3072) and so on.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Loss functions
# ============================================================================
banner("Loss functions: how wrong, and how confidently wrong", "MEDIUM")
step("2.1  One prediction, true answer = class 1")
cases = {"confident and right (p = 0.99)": 0.99, "leaning right (p = 0.7)": 0.7, "unsure (p = 0.5)": 0.5,
         "leaning wrong (p = 0.3)": 0.3, "confident and WRONG (p = 0.01)": 0.01}
rows = {}
for name, p in cases.items():
    logit = np.log(p / (1 - p))
    rows[name] = {"squared error (1-p)^2": (1 - p) ** 2, "cross-entropy -log p": bce_with_logits(np.array([logit]), np.array([1]))[0],
                  "accuracy (p > 0.5)": int(p > 0.5)}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""Squared error tops out at 1. Cross-entropy grows without limit as the model
becomes confidently wrong: 4.6 for p = 0.01, 0.01 for p = 0.99. That is what
classification wants: being sure and wrong is the worst outcome. Accuracy
only sees 0 or 1 and can't tell 'barely right' from 'certain'.""")

step("2.2  Numerical traps, and the stable formulas")
big = np.array([-800.0])
with np.errstate(all="ignore"):
    naive = -np.log(1 / (1 + np.exp(-big)))
    naive_soft = np.exp([1000.0, 999.0]) / np.exp([1000.0, 999.0]).sum()
show("Naive -log(sigmoid(-800))", float(naive[0]))
show("Stable bce_with_logits(-800, y=1)", float(bce_with_logits(big, np.array([1]))[0]))
show("Naive softmax([1000, 999])", naive_soft.tolist())
show("Stable softmax([1000, 999]) (subtract the max)", softmax(np.array([1000.0, 999.0])).round(4).tolist())
note("""exp(800) overflows a 64-bit float, so the naive formulas return inf and nan,
and one nan poisons every weight in the network. Real libraries compute the
loss from LOGITS with algebra that never takes exp of a big number (the toolkit
does the same). That's why PyTorch's cross_entropy takes logits, not
probabilities.""")

step("2.3  The loss an LLM minimises")
rows = {}
for vocab in [10, 1000, 50_257]:
    uniform = np.zeros((1, vocab))
    rows[f"vocabulary of {vocab:,} tokens"] = {"loss of a uniform guess = ln(V)": softmax_ce(uniform, np.array([0]))[0],
                                              "perplexity = exp(loss)": vocab}
print(pd.DataFrame(rows).T.round(3).to_string())
three = np.array([[2.0, 1.0, 0.1]])
show("Softmax of logits [2, 1, 0.1]", softmax(three).round(3).tolist())
show("Cross-entropy if the true class is 0 / 2", f"{softmax_ce(three, np.array([0]))[0]:.3f} / {softmax_ce(three, np.array([2]))[0]:.3f}")
note("""An LLM outputs one logit per vocabulary token (GPT-2 has 50,257), softmax
turns them into next-token probabilities, and the loss is -log P(the token
that actually came next). Guessing uniformly costs ln(50,257) = 10.8.
'Perplexity' is exp(loss): how many tokens the model is effectively choosing
between. Training an LLM is driving this one number down, trillions of times.""")

step("2.4  Regression uses mean squared error")
pred, truth = np.array([210.0, 340.0, 125.0]), np.array([200.0, 300.0, 130.0])
show("MSE for house prices (k$) [210,340,125] vs [200,300,130]", mse(pred, truth)[0])
note("""For numbers, the loss is the squared error of Module 3. The network is the
same; only the last layer and the loss change with the kind of answer.""")

# %% =========================================================================
# LEVEL 3 (HARD): Why we train on loss, not accuracy
# ============================================================================
banner("Accuracy can't be trained; the loss is its smooth stand-in", "HARD")
X, y = make_spirals(n=2000)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
spiral = MLP([2, 32, 32, 1], activation="tanh", loss="bce", seed=0).fit(X_tr, y_tr, epochs=15, batch_size=32, optimizer=Adam(0.01))
show("A network stopped early on purpose: test accuracy", spiral.score(X_te, y_te))
step("3.1  Nudge one weight a little: what moves?")
grads = spiral.backward(spiral.loss_and_grad(spiral.forward(X_tr), y_tr)[1])
k = np.unravel_index(np.abs(grads[2]).argmax(), grads[2].shape)       # the output weight the loss is most sensitive to
rows = {}
w0 = spiral.W[2][k]
for delta in [-0.05, -0.02, 0.0, 0.02, 0.05]:
    spiral.W[2][k] = w0 + delta
    rows[f"{delta:+.2f}"] = {"train loss": spiral.loss_and_grad(spiral.forward(X_tr), y_tr)[0], "train accuracy": spiral.score(X_tr, y_tr)}
spiral.W[2][k] = w0
nudge = pd.DataFrame(rows).T
nudge.index.name = "change to one weight"
print(nudge.to_string(float_format=lambda v: f"{v:.6f}"))
note("""The loss falls steadily from left to right: its slope says 'increase this
weight'. Accuracy moves in jumps of 1/1,400 (one training point flipping) and
goes up and down: it has no slope to follow, and a tiny enough nudge doesn't
change it at all. The loss changes smoothly with every weight; its slope is
the GRADIENT, and Lab 04 computes it for every weight at once. We optimise
the loss and REPORT accuracy (or whatever the business cares about).""")

step("3.2  Same accuracy, very different loss")
rows = {}
for scale in [0.25, 1.0, 4.0, 16.0]:
    W_last, b_last = spiral.W[-1].copy(), spiral.b[-1].copy()
    spiral.W[-1] *= scale
    spiral.b[-1] *= scale
    p = spiral.predict_proba(X_te)
    rows[f"logits x {scale:g}"] = {"test accuracy": spiral.score(X_te, y_te),
                                   "test cross-entropy": bce_with_logits(spiral.forward(X_te), y_te)[0],
                                   "average confidence": np.mean(np.maximum(p, 1 - p))}
    spiral.W[-1], spiral.b[-1] = W_last, b_last
conf = pd.DataFrame(rows).T
print(conf.round(3).to_string())
note(f"""Multiplying the last layer by a constant changes no prediction (the sign of
every logit stays the same), so accuracy is identical in every row. The loss
is not: too timid (x0.25, {conf.iloc[0, 1]:.2f}) and too sure (x16, {conf.iloc[-1, 1]:.2f}) both cost more than
the trained scale ({conf.iloc[1, 1]:.2f}). At x16 the model is {conf.iloc[-1, 2]:.0%} sure on average while
right only {conf.iloc[0, 0]:.0%} of the time, and every mistake becomes expensive. The loss
measures CONFIDENCE, which matters whenever probabilities are used
downstream (Module 4's calibration), and for LLMs, where the probabilities
ARE the output (Lab 13).""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    pp = np.linspace(0.005, 0.995, 200)
    axes[0].plot(pp, -np.log(pp), label="cross-entropy  -log p")
    axes[0].plot(pp, (1 - pp) ** 2, label="squared error  (1-p)^2")
    axes[0].set(xlabel="probability given to the TRUE class", ylabel="loss", title="Confidently wrong costs the most")
    axes[0].legend()
    axes[1].plot(nudge.index.astype(float), nudge["train loss"], "o-")
    ax2 = axes[1].twinx()
    ax2.plot(nudge.index.astype(float), nudge["train accuracy"], "s--", color="C3")
    axes[1].set(xlabel="change to one weight", ylabel="loss (blue)", title="Loss moves smoothly; accuracy (red) doesn't")
    axes[2].bar(conf.index, conf["test cross-entropy"], color="C2")
    axes[2].set(ylabel="test cross-entropy", title=f"Same {conf.iloc[0, 0]:.0%} accuracy, different confidence")
    save_fig(fig, "03_forward_propagation_and_loss")

banner("KEY TAKEAWAYS")
print("""  1. Forward propagation = weighted sums + activations, layer by layer; for a batch, matrix multiplications.
  2. The last layer outputs logits; sigmoid / softmax turn them into probabilities.
  3. Cross-entropy = -log P(true answer): it punishes confident mistakes without limit.
  4. Compute losses from logits (stable formulas); naive exp/log overflows into nan.
  5. LLMs minimise the same cross-entropy over a 50k-token vocabulary; perplexity = exp(loss).
  6. Train on the loss (smooth, has a slope), report the business metric; the loss also measures confidence.""")
