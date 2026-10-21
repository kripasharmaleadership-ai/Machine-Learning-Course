"""
LAB 02 - The Perceptron and the Artificial Neuron
=================================================

ONE NEURON
    inputs x1..xn  ->  weighted sum  z = w1*x1 + ... + wn*xn + b  ->  activation  a = f(z)  ->  output
    weights w      how much each input matters (and in which direction)
    bias b         where the threshold sits: the output when every input is 0
    activation f   step (perceptron, 1958) | sigmoid | tanh | ReLU (most modern networks)

    A sigmoid neuron IS logistic regression (Module 4). A network is many of these, in layers, where the outputs
    of one layer are the inputs of the next.

THE PERCEPTRON LEARNING RULE (Rosenblatt, 1958)
    predict sign(w.x + b); when wrong:  w <- w + lr * y * x,  b <- b + lr * y       (labels y in {-1, +1})
    Guaranteed to stop if a straight line can separate the classes. If not, it never stops.

LEVELS
    EASY   : One neuron, by hand: a loan-approval rule. Then the perceptron learns AND and OR.
    MEDIUM : Activation functions side by side; a sigmoid neuron trained here = sklearn's logistic regression.
    HARD   : XOR breaks the perceptron (the 1969 "AI winter" argument). Two layers fix it, but only WITH an
             activation: stack linear layers without one and you get... one linear layer.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml_lab_utils import HAS_MPL, banner, make_spirals, note, plt, save_fig, show, step
from nn_toolkit import ACTIVATIONS, MLP, Adam, Perceptron, sigmoid

# %% =========================================================================
# LEVEL 1 (EASY): One neuron by hand, then one that learns
# ============================================================================
banner("An artificial neuron is a weighted vote plus a threshold", "EASY")
step("1.1  A hand-made loan neuron")
w = np.array([0.04, -0.9, 0.02])                                     # income_k, late_payments, tenure_months
b = -2.0
applicants = pd.DataFrame({"income_k": [30, 85, 60, 120], "late_payments": [0, 3, 1, 0], "tenure_months": [12, 60, 36, 4]},
                          index=["Ana", "Ben", "Chen", "Dee"])
z = applicants.to_numpy(float) @ w + b
applicants["z = w.x + b"] = z.round(2)
applicants["step(z): approve?"] = np.where(z > 0, "yes", "no")
applicants["sigmoid(z): P(approve)"] = sigmoid(z).round(3)
print(applicants.to_string())
note("""Each input is multiplied by its weight and summed, plus the bias: z. A positive
weight (income) pushes towards 'approve', a negative one (late payments)
pushes away. The bias of -2 means 'start sceptical'. A STEP activation gives a
hard yes/no (the 1958 perceptron); a SIGMOID turns z into a probability. Here
a human chose the weights. Learning means finding them from data.""")

step("1.2  The perceptron learns AND and OR")
X2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
for gate, target in {"AND": [-1, -1, -1, 1], "OR": [-1, 1, 1, 1]}.items():
    p = Perceptron(lr=1.0, epochs=20).fit(X2, np.array(target))
    show(f"{gate}: learned w, b / mistakes per epoch", f"w={p.w.tolist()} b={p.b}   {p.mistakes_}")
    show(f"{gate}: predictions for 00 01 10 11", p.predict(X2).tolist())
note("""The perceptron starts with all weights at zero and nudges them after every
mistake. Within a few epochs it makes zero mistakes and stops. AND needs a
high threshold (a very negative bias), OR a low one: same neuron, different
bias. Each learned rule is a straight line through the 2-D input square.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Activation functions, and the neuron you already know
# ============================================================================
banner("Activation functions, and logistic regression in disguise", "MEDIUM")
step("2.1  The four activations at a few inputs (value / slope)")
zs = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
rows = {name: [f"{f(v):+.3f} / {df(v):.3f}" for v in zs] for name, (f, df) in ACTIVATIONS.items()}
print(pd.DataFrame(rows, index=[f"z={v:+.0f}" for v in zs]).T.to_string())
note("""Sigmoid squashes to (0, 1), tanh to (-1, 1); both FLATTEN at the edges: at
z = 4 the sigmoid's slope is only 0.018. ReLU is max(0, z): slope 1 for any
positive input, 0 otherwise. Learning follows slopes (Lab 04), so flat regions
learn slowly. That is why ReLU (and its cousin GELU, used in GPT) replaced
sigmoid in hidden layers. The slope column is what backpropagation multiplies.""")

step("2.2  A sigmoid neuron trained by our code vs sklearn's logistic regression")
X, y = make_spirals(n=2000)
Xs = X * [1.0, 0.5] + [0.3, 0]                                       # any 2-D data will do
neuron = MLP([2, 1], loss="bce", seed=0).fit(Xs, y, epochs=300, batch_size=2000, optimizer=Adam(0.05))
lr_sk = LogisticRegression(C=1e6).fit(Xs, y)
cmp = pd.DataFrame({"our neuron": np.r_[neuron.W[0].ravel(), neuron.b[0]],
                    "sklearn LogisticRegression": np.r_[lr_sk.coef_.ravel(), lr_sk.intercept_]}, index=["w1", "w2", "b"])
print(cmp.round(4).to_string())
note("""One neuron with a sigmoid, trained to minimise cross-entropy, lands on the same
weights as logistic regression: they are the same model. Everything in
Modules 3-4 (weights, loss, gradient descent) carries straight over. A
neural network is what you get when you stack these and let the inner ones
learn features instead of reading them from a table.""")

# %% =========================================================================
# LEVEL 3 (HARD): XOR, and why the activation matters
# ============================================================================
banner("XOR: the problem that stalled neural networks", "HARD")
y_xor = np.array([-1, 1, 1, -1])
p = Perceptron(lr=1.0, epochs=30).fit(X2, y_xor)
show("XOR: perceptron mistakes per epoch (30 epochs)", p.mistakes_[:12] + ["..."])
show("XOR: predictions vs truth", f"{p.predict(X2).tolist()} vs {y_xor.tolist()}")
note("""XOR is 'one or the other, not both'. No straight line puts (0,1) and (1,0) on
one side and (0,0), (1,1) on the other, so the perceptron NEVER stops making
mistakes. Minsky and Papert's 1969 book made this point, and funding for
neural networks dried up for over a decade.""")

step("3.1  Two layers solve XOR (weights set by hand)")
step_fn = lambda v: (v > 0).astype(int)
h_or = step_fn(X2 @ [1, 1] - 0.5)                                    # hidden neuron 1: OR
h_nand = step_fn(X2 @ [-1, -1] + 1.5)                                # hidden neuron 2: NAND
out = step_fn(h_or + h_nand - 1.5)                                   # output: AND of the two
print(pd.DataFrame({"x1": X2[:, 0].astype(int), "x2": X2[:, 1].astype(int), "hidden OR": h_or,
                    "hidden NAND": h_nand, "output = AND(OR, NAND)": out}).to_string(index=False))
note("""The hidden layer re-describes each input with two new features (OR and NAND).
In that new space XOR IS a straight line: 'both features on'. This is the
core idea of deep learning: each layer builds features that make the next
layer's job linear. In Lab 04, backpropagation finds such weights by itself.""")

step("3.2  Plot twist: depth without an activation is useless")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
rows = {}
for act in ["linear", "relu", "tanh"]:
    for sizes in ([2, 1], [2, 32, 32, 32, 1]):
        if act != "linear" and len(sizes) == 2:
            continue
        m = MLP(sizes, activation=act, loss="bce", seed=0).fit(X_tr, y_tr, epochs=300, batch_size=32, optimizer=Adam(0.01))
        rows[f"{len(sizes) - 2} hidden layers, activation = {act}"] = {"trainable parameters": m.n_params,
                                                                      "spiral test accuracy": m.score(X_te, y_te)}
deep = pd.DataFrame(rows).T
print(deep.to_string(formatters={"trainable parameters": "{:,.0f}".format, "spiral test accuracy": "{:.3f}".format}))
note(f"""Three hidden layers and {deep.iloc[1, 0]:,.0f} parameters, but with a LINEAR activation the
whole network is still one weighted sum: W3(W2(W1 x)) = (W3 W2 W1) x. It draws
the same single straight line as one neuron ({deep.iloc[1, 1]:.0%} vs {deep.iloc[0, 1]:.0%}; the gap is training
noise). Add a non-linear activation (ReLU, tanh)
and the same layers solve the spiral. The activation is the small piece that
makes depth worth having.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    zz = np.linspace(-5, 5, 300)
    for name in ["sigmoid", "tanh", "relu"]:
        axes[0].plot(zz, ACTIVATIONS[name][0](zz), label=name)
        axes[1].plot(zz, ACTIVATIONS[name][1](zz), label=name)
    axes[0].set(title="Activation functions", xlabel="z", ylim=(-1.2, 2.5))
    axes[1].set(title="Their slopes (what backprop multiplies)", xlabel="z")
    for ax in axes[:2]:
        ax.legend()
        ax.axhline(0, color="grey", lw=0.5)
    axes[2].scatter(*X2.T, c=["C0" if v < 0 else "C3" for v in y_xor], s=300)
    for (a, b_), v in zip(X2, y_xor):
        axes[2].annotate("1" if v > 0 else "0", (a, b_), ha="center", va="center", color="white", weight="bold")
    axes[2].set(title="XOR: no single line separates red from blue", xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xlabel="x1", ylabel="x2")
    save_fig(fig, "02_perceptron_and_artificial_neuron")

banner("KEY TAKEAWAYS")
print("""  1. A neuron = weighted sum of inputs + bias, passed through an activation.
  2. Weights say how much each input matters; the bias sets the threshold.
  3. The perceptron rule learns any straight-line rule, and loops forever when none exists (XOR).
  4. A sigmoid neuron is logistic regression; networks are layers of them learning their own features.
  5. Without a non-linear activation, any number of layers collapses to one. ReLU/GELU make depth work.""")
